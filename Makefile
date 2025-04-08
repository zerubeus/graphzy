# Compiler: Use clang on macOS
CC = clang

# Executable name
TARGET = graphzy

# Source files
SRCS = main.c

# Object files
OBJS = $(SRCS:.c=.o)

# Compiler flags
CFLAGS = -Wall -Wextra -g -O2

# Build directory
BUILD_DIR = build

# --- Raylib flags and paths ---
# macOS with Homebrew - direct path detection
BREW_RAYLIB_PREFIX := $(shell brew --prefix raylib 2>/dev/null)

ifdef BREW_RAYLIB_PREFIX
    CFLAGS += -I$(BREW_RAYLIB_PREFIX)/include
    LDFLAGS += -L$(BREW_RAYLIB_PREFIX)/lib -lraylib
else
    # Fallback to pkg-config if brew command fails
    PKG_CONFIG_EXISTS := $(shell command -v pkg-config 2> /dev/null)
    ifdef PKG_CONFIG_EXISTS
        CFLAGS += $(shell pkg-config --cflags raylib 2>/dev/null)
        LDFLAGS += $(shell pkg-config --libs raylib 2>/dev/null)
    else
        # Last resort: try common paths
        CFLAGS += -I/opt/homebrew/include
        LDFLAGS += -L/opt/homebrew/lib -lraylib
    endif
endif

# macOS-specific frameworks
ifeq ($(shell uname), Darwin)
    LDFLAGS += -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL
endif

# Add math library
LDFLAGS += -lm

# Show environment info for debugging
info:
	@echo "CFLAGS = $(CFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"
	@echo "BREW_RAYLIB_PREFIX = $(BREW_RAYLIB_PREFIX)"

# Default rule: Build the executable
all: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Linking rule
$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(BUILD_DIR)/$(TARGET) $(LDFLAGS)

# Compilation rule for .c files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to run the executable with optional file argument
run: all
	@if [ -z "$(FILE)" ]; then \
		./$(BUILD_DIR)/$(TARGET); \
	else \
		./$(BUILD_DIR)/$(TARGET) $(FILE); \
	fi

# Rule to clean up object files and the executable
clean:
	rm -f $(OBJS)
	rm -rf $(BUILD_DIR)

# Phony targets avoid conflicts with files named 'all', 'run', 'clean'
.PHONY: all run clean info 