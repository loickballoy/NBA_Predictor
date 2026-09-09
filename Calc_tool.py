from __future__ import annotations

import concerto as conc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, cast

#
# 1. configuration
#


@dataclass
class Config:
	deriv_threshold: float = 0.02
	smooth_window_s: float = 10.0
	min_phase_duration_s: float = 60.0
	low_pct: float = 0.1
	high_pct: float = 0.9
	settle_pct: float =0.1
	
	temp_tolerance_C: float = 2.5
	debit_tolerance_pct: float = 0.1
	debit_deadtime_pct: float = 0.02

#
# 2. Phases
#

@dataclass
class Phase:
	kind: str
	level: float
	start_idx: int
	end_idx: int
	cycle: int = 0
	
	def t_start(self, t):
		return t[self.start_idx]
		
	def t_end(self, t):
		return t[self.end_idx]

def _smooth(y: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
	dt = np.median(np.diff(t))
	win = max(3, int(round(window_s / dt)))
	if win % 2 == 0:
		win += 1
	kernel = np.ones(win) / win
	return np.convolve(y, kernel, mode="same")
	
def _interpolate_nans(t, y, label):
	y = np.asarray(y, dtype=float)
	mask_nan = np.isnan(y)
	if not mask_nan.any():
		return y
	n_nan = int(mask_nan.sum())
	print(f"[WARNING] {n_nan} found in {label}")
	y_clean = y.copy()
	valid = ~mask_nan
	y_clean[mask_nan] = np.interp(t[mask_nan], t[valid], y[valid])
	return y_clean
	
def _find_recross_idx(t, actual, target, sign, start_idx):
	seg = actual[start_idx:]
	reached = sign * seg >= sign * target
	if not reached.any():
		return None
	idx_first_reach = np.argmax(reached)
	after = seg[idx_first_reach:]
	recrossed = sign * after < sign * target
	if not recrossed.any():
		return None
	idx_recross_rel = np.argmax(recrossed)
	return start_idx + idx_first_reach + idx_recross_rel
	
def segment_phases(t: np.ndarray, consigne: np.ndarray, cfg: Config) -> list[Phase]:
	dt = float(np.median(np.diff(t)))
	amp_threshold = cfg.deriv_threshold * dt
	
	raw_diff = np.abs(np.diff(consigne))
	in_change = raw_diff > amp_threshold
	
	transitions: list[tuple[int, int]] = []
	i = 0
	while i < len(in_change):
		if in_change[i]:
			j = i
			while j < len(in_change) and in_change[j]:
				j+=1
			transitions.append((i, j))
			i = j
		else:
			i += 1
	
	if not transitions:
		ph = Phase(kind="palier", level=float(np.nanmedian(consigne)), start_idx=0, end_idx=len(consigne) -1)
		ph.cycle = 1
		return [ph]
		
	phases: list[Phase] = []
	
	if transitions[0][0] > 0:
		phases.append(Phase(kind="palier", level=float(np.nanmedian(consigne[:transitions[0][0]])), start_idx=0, end_idx=transitions[0][0] - 1))
	
	for k, (ts, te) in enumerate(transitions):
		amp = float(consigne[te]) - float(consigne[ts])
		if abs(amp) > amp_threshold:
			kind = "rampe_montee" if amp > 0 else "rampe_descente"
			phases.append(Phase(kind=kind, level=None, start_idx=ts, end_idx=te))
		next_ts = transitions[k + 1][0] if k + 1 < len(transitions) else len(consigne)
		i1_pal = next_ts - 1 if k + 1 < len(transitions) else len(consigne) - 1
		if te <= t1_pal:
			phases.append(Phase(kind="palier", level=float+(np.nanmedian(consigne[te:i1_pal + 1])), start_idx=te, end_idx=i1_pal))
			
	merged: list[Phase] = []
	for ph in phases:
		duration = t[ph.end_idx] - t[ph.start_idx]
		if merged and duration < cfg.min_phase_duration_s:
			prev = merged[-1]
			prev.end_idx = ph.end_idx
		else:
			merged.append(ph)
			
	final: list[Phase] = []
	for ph in merged:
		if final and final[-1].kind == ph.kind and (ph.kind != "palier" or np.isclose(final[-1].level, ph.level, atol=1.0)):
			final[-1].end_idx = ph.end_idx
		else:
			final.append(ph)
			
	counters: dict[str, int] = {}
	for ph in final:
		if ph.kind == "palier" and (ph.level is None or np.isnan(ph.level)):
			ph.level = float(np.nanmedian(consigne))
		key = ph.kind if ph.kind != "palier" else f"palier_{round(ph.level)}"
		counters[key] = counters.get(key, 0) + 1
		ph.cycle = counters[key]
		
	return final
	
def _find_next_transition_start(phases: list[Phase], idx: int) -> int | None:
		for later in phases[idx + 1:]:
			if later.kind != "palier":
				return later.start_idx
		return None
		
def _settling_time(seg_t: np.ndarray, seg_values: np.ndarray, target: float, tolerance: float, t_ref: float) -> float:
	if len(seg_values) == 0:
		return np.nan
	in_band = np.abs(seg_values - target) <= tolerance
	if in_band.all():
		idx_settle = 0
	elif not in_band.any():
		return np.nan
	else:
		out_idx = np.where(~in_band)[0]
		idx_settle = out_idx[-1] + 1
		if idx_settle >= len(seg_values):
			return np.nan
			
	return float(seg_t[idx_settle] - t_ref)
	
def _dead_time(seg_t: np.ndarray, seg_values: np.ndarray, start_value: float, end_value: float, sign: float, pct_threshold: float, t_ref: float) -> float:
	amp = end_value - start_value
	if len(seg_values) == 0 or amp == 0:
		return np.nan
	sign = float(np.sign(amp))
	threshold = start_value + sign * pct_threshold * abs(amp)
	crossed = sign* seg_values >= sign * threshold
	if not crossed.any():
		return np.nan
	return float(seg_t[np.argmax(crossed)] - t_ref)
	

#
# 3. Analyse d'une rampe
#	
	
def analyze_ramp(t, consigne, actual, debit, consigne_debit, phase: Phase, cfg: Config, search_end_idx: int | None = None, post_window_s: float = 300.0) ->dict:
	i0, i1 = phase.start_idx, phase.end_idx
	c0, c1 = consigne[i0], consigne[i1]
	a0, a1 = actual[i0], actual[i1]
	
	if c0 == c1:
		return
		
	amp = c1 - c0
	sign = np.sign(amp) if amp != 0 else 1
	
	thr_low = c0 + cfg.low_pct * amp
	thr_high = c0 + cfg.high_pct * amp
	
	th_mesured_low = a0+ cfg.low_pct * amp
	th_mesured_high = a0 + cfg.high_pct * amp
	
	
	thr_delay = c0 + sign * 1.0
	
	t_consigne_start = t[i0]
	
	if search_end_idx is not None:
		end = min(len(t), search_end_idx)
	else:
		dt = np.median(np.diff(t))
		n_post = int(post_window_s / dt)
		end = min(len(t), i1 + n_post)
		
	seg_actual = actual[i0:end]
	seg_t = t[i0:end]
	
	idx_cross_delay = np.argmax(sign * seg_actual >= sign * thr_delay) if np.any(sign * seg_actual >= sign * thr_delay) else None
	delay_s = (seg_t[idx_cross_delay] - t_consigne_start) if idx_cross_delay is not None else np.nan
	
	idx_cross_low = np.argmax(sign * seg_actual >= sign * thr_low) if np.any(sign * seg_actual >= sign * thr_low) else None
	idx_cross_high = np.argmax(sign * seg_actual >= sign * thr_high) if np.any(sign * seg_actual >= sign * thr_high) else None
	
	t_low = seg_t[idx_cross_low] if idx_cross_low is not None else np.nan
	t_high = seg_t[idx_cross_high] if idx_cross_high is not None else np.nan
	gradient = (cfg.high_pct - cfg.low_pct) * (a1 - a0) / (t_high - t_low) if (t_high - t_low) != 0 else np.nan
	consigne_gradient = (cfg.high_pct - cfg.low_pct) * (c1 - c0) / (t_high - t_low) if (t_high - t_low) != 0 else np.nan
	
	seg_post = actual[i1:end]
	seg_post_t = t[i1:end]
	
	crossed = sign * seg_post >= sign * c1
	if crossed.any():
		idx_first_cross = np.argmax(crossed)
		after_cross = seg_post[idx_first_cross:]
		idx_ov_rel = np.argmax(after_cross) if sign > 0 else np.argmin(after_cross)
		idx_ov = idx_first_cross + idx_ov_rel
		overshoot = seg_post[idx_ov] - c1
	else:
		overshoot = np.nan
		
	in_band = np.abs(seg_post-c1) <= cfg.temp_tolerance_C
	if in_band.all():
		idx_settle = 0
	elif not in_band.any():
		idx_settle = None
	else:
		out_idx = np.where(~in_band)[0]
		idx_settle = out_idx[-1] + 1
		if idx_settle >= len(seg_post):
			idx_settle = None
			
	response_time = (seg_post_t[idx_settle] - t[i1]) if idx_settle is not None else np.nan
	
	seg_d_ramp = debit[i0:i1 + 1]
	if np.isscalar(consigne_debit):
		seg_cd_ramp = np.full_like(seg_d_ramp, consigne_debit, dtype=float)
	else:
		seg_cd_ramp = consigne_debit[i0:i1+1]
	with np.errstate(divide="ignore", invalid="ignore"):
		desadapt_ramp_pct = np.where(seg_cd_ramp !=0, (seg_d_ramp - seg_cd_ramp) / seg_cd_ramp * 100, np.nan)
		
	
	return {
		"cycle": phase.cycle,
		"kind": phase.kind,
		"consigne_depart": c0,
		"consigne_arrivee": c1,
		"tps_de_retard_s": round(delay_s, 1) if not np.isnan(delay_s)else np.nan,
		"1er_depassement": round(overshoot, 2) if not np.isnan(overshoot) else np.nan,
		"tps_de_reponse_s": round(response_time, 2) if not np.isnan(response_time) else np.nan,
		"gradient_consigne_C": round(consigne_gradient, 4) * 60 if not np.isnan(consigne_gradient) else np.nan,
		"gradient_reel_C": round(gradient, 4) * 60 if not np.isnan(gradient) else np.nan,
		"plage_gradient_s": (round(t_low, 0), round(t_high, 0)) if not np.isnan(t_low) else None,
		"ecart_debit_min_pct": round(np.nanmin(desadapt_ramp_pct), 2) if not np.all(np.isnan(desadapt_ramp_pct)) else np.nan,
		"ecart_debit_max_pct": round(np.nanmax(desadapt_ramp_pct), 2) if not np.all(np.isnan(desadapt_ramp_pct)) else np.nan
	}
	
#
# 4. Analyse d'un palier
#

def analyze_plateau(t, consigne_temp, temp_mesuree, debit, consigne_debit, phase: Phase, cfg: Config) -> dict:
	i0, i1 = phase.start_idx, phase.end_idx
	mid = (i0 + i1) // 2

	seg_second_half = temp_mesuree[mid:i1 + 1]
	consigne_val = phase.level
	print(consigne_val)
	
	seg_d = debit[mid: i1+1]
	if np.isscalar(consigne_debit):
		seg_cd = np.full_like(seg_d, consigne_debit, dtyp=float)
	else:
		seg_cd = consigne_debit[mid:i1+1]
	with np.errstate(divide="ignore", invalid="ignore"):
		desadapt_pct = np.where(seg_cd != 0, (seg_d - seg_cd) / seg_cd * 100, np.nan)
	
	return {
		"cycle": phase.cycle,
		"consigne_C": consigne_val,
		"ecart_min": round(seg_second_half.min() - consigne_val, 2),
		"ecart_max": round(seg_second_half.max() - consigne_val, 2),
		"ecart_debit_min_pct": round(np.nanmin(desadapt_pct), 2) if not np.all(np.isnan(desadapt_pct)) else np.nan,
		"ecart_debit_max_pct": round(np.nanmax(desadapt_pct), 2) if not np.all(np.isnan(desadapt_pct)) else np.nan,
		"plage_s": (round(t[i0], 0), round(t[i1], 0)),
		"plage_2e_moitie_s": (round(t[mid], 0), round(t[i1],0)),
	}
	
#
# 5. Orchestation
#

def run_full_analysis(consigne_temp, temp_mesuree, debit, consigne_debit, t: np.ndarray | None = None, cfg: Config | None = None):
	cfg = cfg or Config()
	if t is None:
		t = np.arange(len(consigne_temp), dtype=float)
		
	consigne_temp = _interpolate_nans(t, consigne_temp, "consigne_temp")
	temp_mesuree = _interpolate_nans(t, temp_mesuree, "temp_mesuree")
	debit = _interpolate_nans(t, debit, "debit")
	if not np.isscalar(consigne_debit):
		consigne_debit = _interpolate_nans(t, consigne_debit, "consigne_debit")
	
	
	phases = segment_phases(t, consigne_temp, cfg)
	print(phases)
	
	ramp_rows = []
	plateau_rows = []
	for idx, ph in enumerate(phases):
		if ph.kind in ("rampe_montee", "rampe_descente"):
			next_transition_idx = _find_next_transition_start(phases, phases.index(ph))
			res = analyze_ramp(t, consigne_temp, temp_mesuree, debit, consigne_debit, ph, cfg, search_end_idx = next_transition_idx)
			if res is not None:
				ramp_rows.append(res)
		else:
			plateau_rows.append(analyze_plateau(t, consigne_temp, temp_mesuree, debit, consigne_debit, ph, cfg))
			
	df_ramps = pd.DataFrame(ramp_rows)
	df_plateaus = pd.DataFrame(plateau_rows)
	
	tables = {}
	
	if not df_ramps.empty:
		for kind in df_ramps["kind"].unique():
			sub = df_ramps[df_ramps["kind"] == kind].set_index("cycle")
			tables[kind] = sub.drop(columns=["kind"]).T
	if not df_plateaus.empty:
		for level, grp in df_plateaus.groupby("consigne_C"):
			sub = grp.set_index("cycle")
			tables[f"palier_{level}C"] = sub.T
	print(tables)
	
	return t, phases, df_ramps, df_plateaus, tables
	
	
#
# ISO TEMP
#

def segment_by_debit_steps(t: np.ndarray, consigne_debit: np.ndarray) -> list[tuple[int, int]]:
	diffs = np.diff(consigne_debit)
	change_idx = np.where(np.abs(diffs) > 1e-9)[0] + 1
	boundaries = [0] + list(change_idx) + [len(consigne_debit)]
	return [(boundaries[i], boundaries[i + 1] - 1) for i in range(len(boundaries) - 1) if boundaries[i + 1] - 1 >= boundaries[i]]
	
def analyze_isoT_phase(t: np.ndarray, consigne_temp, temp_mesuree, debit, consigne_debit, i0, i1, phase_num, cfg: Config) -> dict:
	q_len = max(1, (i1 - i0) // 4)
	q1_end = min(i0 + q_len, i1)
	
	consigne_temp_val = float(np.nanmedian(consigne_temp[i0:i1+1]))
	c1_debit = float(consigne_debit[i1])
	c0_debit = float(consigne_debit[i0 - 1]) if i0 > 0 else c1_debit
	sign_debit = float(np.sign(c1_debit - c0_debit)) if c1_debit != c0_debit else 0.0
	t_ref = float(t[i0])
	
	seg_t_q1 = t[i0:q1_end]
	seg_t_full = t[i0:i1+1]
	seg_t_rest = t[q1_end: i1+1]
	
	seg_temp_q1 = temp_mesuree[i0:q1_end]
	seg_temp_full = temp_mesuree[i0:i1+1]
	seg_temp_rest = temp_mesuree[q1_end: i1+1]
	
	cd = np.asarray(consigne_debit, dtype=float)
	seg_cd_q1 = cd[i0:q1_end]
	seg_cd_full = cd[i0:i1+1]
	seg_cd_rest = cd[q1_end: i1+1]
	
	seg_d_q1 = debit[i0:q1_end]
	seg_d_full = debit[i0:i1+1]
	seg_d_rest = debit[q1_end: i1+1]
	
	err_temp_q1 = seg_temp_q1 - consigne_temp_val
	err_temp_rest = seg_temp_rest - consigne_temp_val
	
	def _dp(sd, scd):
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.where(scd !=0, (sd - scd) / scd * 100, np.nan)
			
	dp_q1 = _dp(seg_d_q1, seg_cd_q1)
	dp_rest = _dp(seg_d_rest, seg_cd_rest)
	
	temps_stab = _settling_time(seg_t_full, seg_temp_full, consigne_temp_val, cfg.temp_tolerance_C, t_ref)
	temps_mort = _dead_time(seg_t_full, seg_d_full, c0_debit, c1_debit, sign_debit, cfg.debit_deadtime_pct, t_ref)
	
	tol_abs = cfg.debit_tolerance_pct * abs(c1_debit) if c1_debit != 0 else np.nan
	temps_rep = _settling_time(seg_t_full, seg_d_full, c1_debit, tol_abs, t_ref)
	
	def sm(a): return round(float(a.min()), 2) if len(a) else np.nan
	def sx(a): return round(float(a.max()), 2) if len(a) else np.nan
	def nm(a): return round(float(np.nanmin(a)), 2) if len(a) and not np.all(np.isnan(a)) else np.nan
	def nx(a): return round(float(np.nanmax(a)), 2) if len(a) and not np.all(np.isnan(a)) else np.nan
	
	return {
		"phase_num": phase_num,
		"consigne_temp_C": round(consigne_temp_val, 2),
		"consigne_debit": round(c1_debit, 1),
		"plage_s": (round(t_ref, 0), round(float(t[i1]), 0)),
		"plage_q1_s": (round(t_ref, 0), round(float(t[q1_end - 1]), 0)) if q1_end > i0 else None,
		"q1_ecart_temp_min": sm(err_temp_q1),
		"q1_ecart_temp_max": sx(err_temp_q1),
		"q1_ecart_debit_min_pct": nm(dp_q1),
		"q1_ecart_debit_max_pct": nx(dp_q1),
		"temps_stabilisation_temp_s": round(temps_stab, 2) if not np.isnan(temps_mort) else np.nan,
		"temps_mort_debit_s": round(temps_mort, 2) if not np.isnan(temps_mort) else np.nan,
		"temps_reponse_debit_s": round(temps_rep, 2) if not np.isnan(temps_rep) else np.nan,
		"qreste_ecart_temp_min": sm(err_temp_rest),
		"qreste_ecart_temp_max": sx(err_temp_rest),
		"qreste_ecart_debit_min_pct": nm(dp_rest),
		"qreste_ecart_debit_max_pct": nx(dp_rest)
	}
	
def run_isoT_analysis(consigne_temp, temp_mesuree, debit, consigne_debit, t: np.ndarray | None = None, cfg: Config | None = None):
	cfg =cfg or Config()
	if t is None:
		t = np.arange(len(consigne_temp), dtype=float)
	
	consigne_temp = _interpolate_nans(t, consigne_temp, "consigne_temp")
	temp_mesuree = _interpolate_nans(t, temp_mesuree, "temp_mesuree")
	debit = _interpolate_nans(t, debit, "debit")
	consigne_debit = _interpolate_nans(t, consigne_debit, "consigne_debit")

	phase_bounds = segment_by_debit_steps(t, consigne_debit)
	
	results = [analyze_isoT_phase(t, consigne_temp, temp_mesuree, debit, consigne_debit, i0, i1, num, cfg) for num, (i0, i1) in enumerate(phase_bounds, start=1)]
	
	return t, phase_bounds, results
	
def build_isoT_matrix(results):
	results = results[1:]
	n = len(results)
	header = [""] + [f"Ph{res['phase_num']} ({res['consigne_debit']}" for res in results]
	rows = [header]
	
	for label in ["consigne_temp_C", "consigne_debit", "plage_s", "plage_q1_s"]:
		rows.append([label] + [str(res.get(label, "")) for res in results])
	rows.append([""] + [""] * n)
		
	rows.append(["-- 1er Q --"] + [""] * 2 * n)
	for label in [
		"q1_ecart_temp_min", "q1_ecart_temp_max",
		"q1_ecart_debit_min_pct", "q1_ecart_debit_max_pct",
		"temps_stabilisation_temp_s", "temps_mort_debit_s", "temps_reponse_debit_s"
	]:
		rows.append([label] + [str(res.get(label, "")) for res in results])
	rows.append([""] + [""] * n)
	
	rows.append(["-- Reste --"] + [""] * 2 * n)
	for label in [
		"qreste_ecart_temp_min", "qreste_ecart_temp_max",
		"qreste_ecart_debit_min_pct", "qreste_ecart_debit_max_pct"
	]: 
		rows.append([label] + [str(res.get(label, "")) for res in results])
	
	rows.append([""] + [""] * n)
	rows.append([""] + [""] * n)
	return rows

def print_df(df, title: str | None = None):
	if title:
		print(title)
	if df is None:
		print("(aucune donné)")
		return
	
	for line in df.to_string().split("\n"):
		print(line)
			
			
#display

def build_chronological_matrix(t, consigne_temp, temp_mesuree, debit, consigne_debit, phases, cfg):
	rows: list[list[str]] = []
	
	for idx, ph in enumerate(phases):
		kind = ph.kind
		is_ramp = kind in ("rampe_montee", "rampe_descente")
		suffix = f"({ph.level}°C)" if not is_ramp and ph.level is not None else ""
		# if is_ramp and consigne_temp[ph.start_idx] != consigne_temp[ph.end_idx]:
		rows.append([f"-- {kind}{suffix} --", ""])
		
		if is_ramp:
			next_transition_idx = _find_next_transition_start(phases, idx)
			res = analyze_ramp(t, consigne_temp, temp_mesuree, debit, consigne_debit, ph, cfg, search_end_idx=next_transition_idx)
			if res is None:
				continue
			field_labels = [
				"consigne_depart", "consigne_arrivee", "tps_de_retard_s",
				"1er_depassement", "tps_de_reponse_s", "gradient_consigne_C", "gradient_reel_C",
				"plage_gradient_s", "ecart_debit_min_pct", "ecart_debit_max_pct"
			]
		else:
			print(ph)
			res = analyze_plateau(t, consigne_temp, temp_mesuree, debit, consigne_debit, ph ,cfg)
			field_labels = [
				"consigne_C", "ecart_min", "ecart_max",
				"ecart_debit_min_pct", "ecart_debit_max_pct",
				"plage_s", "plage_2e_moitie_s"
			]
			
		for label in field_labels:
			rows.append([label, str(res.get(label, ""))])
			
		rows.append(["", ""])
	rows.append(["", ""])
	rows.append(["", ""])
	header = ["Phase", "Valeur"]
	return [header] + rows
	
def write_matrix_to_grid_table(grid_table, matrix):
	n_rows = len(matrix)
	n_cols = max(len(r) for r in matrix)
	grid_table.row_count = n_rows
	
	cells = grid_table.cells
	for row in range(n_rows):
		row_data = matrix[row]
		for col in range(n_cols):
			value = row_data[col] if col < len(row_data) else ""
			cells[row][col].text = value
			
			
#formula code must be in function f()

def f():
	win = conc.gui.get_active_window()
	input_table = win.get_objects(name="Input Table")[0]
	
	consigne_temp_name = input_table.cells[1][0].text
	mesured_temp_name = input_table.cells[1][1].text
	consigne_debit_name = input_table.cells[1][2].text
	mesured_debit_name = input_table.cells[1][3].text
 
	consigne_temp = conc.ds(consigne_temp_name)
	temp_mesuree = conc.ds(mesured_temp_name)
	debit = conc.ds(mesured_debit_name)
	consigne_debit = conc.ds(consigne_debit_name)
	
	cfg = Config(
		temp_tolerance_C = 2.5,
		debit_tolerance_pct = 0.1
	)
	is_isoT = bool(int(input_table.cells[1][4].text))
	debug_string = "Il s'agit de l'analyse d'une isoTempérature" if is_isoT else "Il s'agit de l'analyse d'un isoDébit"
	print(debug_string)
	
	out = [0]*len(consigne_temp)
	c: int = 0
	
	if is_isoT:
		t, phase_bounds, results = run_isoT_analysis(consigne_temp, temp_mesuree, debit, consigne_debit, cfg=cfg)
		matrix = build_isoT_matrix(results)
		
		for ph in phase_bounds:
			for i in range(ph[0], ph[1]):
				out[i] = c
			c += 10
	else:
		t, phases, df_ramps, df_plateaus, tables = run_full_analysis(consigne_temp, temp_mesuree, debit, consigne_debit, cfg=cfg)
		matrix = build_chronological_matrix(t, consigne_temp, temp_mesuree, debit, consigne_debit, phases, cfg)
		
		for ph in phases:
			for i in range(ph.start_idx, ph.end_idx):
				out[i] = c
			c+= 10
			
	
	
	
		
	grid_table = win.get_objects(name="Result_Table")[0]
	grid_table = cast(conc.gui.forms.GridTable, grid_table)
	
	for i in range(grid_table.row_count):
		for j in range(len(grid_table.cells[0])):
			grid_table.cells[i][j].text = ""
	
	write_matrix_to_grid_table(grid_table, matrix)
	
	return out


