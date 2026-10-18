CXX			:= g++
CXXFLAGS	?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS	+= -Iinclude -MMD -MP
LDFLAGS		?=

BUILD	?= build
BIN		?= bin

LIB_SRC		:= $(wildcard src/*.cpp)
LIB_OBJ		:= $(LIB_SRC:%.cpp=$(BUILD)/%.o)
APPS		:= xor nba
TARGETS		:= $(APPS:%=$(BIN)/%)

.PHONY: all debug run-xor run-nba clean

.SECONDARY:

all: $(TARGETS)

$(BIN)/%: $(BUILD)/apps/%.o $(LIB_OBJ)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $^ -o $@

debug:
	$(MAKE) BUILD=build/debug BIN=bin/debug \
	  CXXFLAGS="-std=c++17 -O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined" \
	  LDFLAGS="-fsanitize=address,undefined"
 
run-xor: $(BIN)/xor
	./$(BIN)/xor --live
 
run-nba: $(BIN)/nba
	./$(BIN)/nba ../collector/data/processed
 
clean:
	rm -rf build bin

-include $(LIB_OBJ:.o=.d) $(APPS:%=$(BUILD)/apps/%.d)