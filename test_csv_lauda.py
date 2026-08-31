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
	
	
	consigne_s = _smooth(consigne, t, cfg.smooth_window_s)
	dcdt = np.gradient(consigne_s, t)
	seg_dcdt = dcdt[607: 1331]

	
	raw_kind = np.where(dcdt > cfg.deriv_threshold, "rampe_montee",
				np.where(dcdt < -cfg.deriv_threshold, "rampe_descente", "palier"))
	
	phases: list[Phase] = []
	start = 0
	for i in range(1, len(raw_kind) + 1):
		if i == len(raw_kind) or raw_kind[i] != raw_kind[start]:
			level = np.nanmedian(consigne[start:i]) if raw_kind[start] == "palier" else None
			if level is not None and np.isnan(level):
				level = np.nanmedian(consigne)
			phases.append(Phase(kind=raw_kind[start], level=level, start_idx=start, end_idx=i-1))
			start = i
	
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
		print(ph)
		key = ph.kind if ph.kind != "palier" else f"palier_{round(ph.level)}"
		counters[key] = counters.get(key, 0) + 1
		ph.cycle = counters[key]
		
	return final	
	
	
def _find_next_transition_start(phases: list[Phase], idx: int) -> int | None:
		for later in phases[idx + 1:]:
			if later.kind != "palier":
				return later.start_idx
		return None
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
	
	t, phases, df_ramps,  df_plateaus, tables = run_full_analysis(consigne_temp, temp_mesuree, debit, consigne_debit, cfg=cfg)
	
	pd.set_option("display.width", 100)
	pd.set_option("display.max_columns", None)
	
	print("=== Phases détectées ===")
	for ph in phases:
		print(f"{ph.kind:15s} cycle {ph.cycle}  [{t[ph.start_idx]:.0f}s -> {t[ph.end_idx]:.0f}s] niveau={ph.level}")
	
	print()
	print_df(tables.get("rampe_montee"), title= "=== ANALYSE TEMP MONTEE ===")
	print()
	print_df(tables.get("rampe_descente"), title= "=== ANALYSE TEMP DESCENTE ===")
	
	print("\n=== temp debit ===")
	for k, v in tables.items():
		if k.startswith("palier"):
			print()
			print_df(v, title=f"-- {k} --")
			
	out = [0]*len(consigne_temp)
	c: int = 0
	
	for ph in phases:
		for i in range(ph.start_idx, ph.end_idx + 1):
			out[i] = c
		c+= 10
		
	grid_table = win.get_objects(name="Result_Table")[0]
	grid_table = cast(conc.gui.forms.GridTable, grid_table)
	matrix = build_chronological_matrix(t, consigne_temp, temp_mesuree, debit, consigne_debit, phases, cfg)
	
	write_matrix_to_grid_table(grid_table, matrix)
	
	return out


