# Compiler: clang is default on macOS, gcc works if installed
CC = clang
# Executable name
TARGET = fenetre
# Source files
SRCS = main.c
# Object files
OBJS = $(SRCS:.c=.o)

# --- Raylib flags using pkg-config (Recommended) ---
# Ensure pkg-config is installed (brew install pkg-config)
# Ensure PKG_CONFIG_PATH is set for Homebrew (check .zshrc/.bash_profile)
# CFLAGS = -Wall -Wextra -g -O2 $(shell pkg-config --cflags raylib)
# LDFLAGS = $(shell pkg-config --libs raylib) -lm

# --- Fallback Manual Flags for Raylib (Hardcoded for /opt/homebrew) ---
# BREW_PREFIX = /opt/homebrew # Adjust if Homebrew is installed elsewhere (REMOVED)
CFLAGS = -Wall -Wextra -g -O2 -I/opt/homebrew/include
# LDFLAGS = -L$(BREW_PREFIX)/lib -lraylib -lm (REMOVED variable use)
# For macOS, Raylib often needs framework flags:
LDFLAGS = -L/opt/homebrew/lib -lraylib -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL -lm
# --------------------------------------------------------------------------

# Default rule: Build the executable
all: $(TARGET)

# Linking rule
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compilation rule for .c files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to run the executable
run: all
	./$(TARGET)

# Rule to clean up object files and the executable
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets avoid conflicts with files named 'all', 'run', 'clean'
.PHONY: all run clean 