# Makefile for Movie Trailer Generation
# Targets: setup, all_trailers, clean

PYTHON = python3
VENV = venv
PIP = $(PYTHON) -m pip
PY_EXEC = $(PYTHON)

# Environment variables
export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCH_LOGS=recompiles,graph_breaks

RUST_BIN = ./assemble_video_rust

all: assets video

setup:
	$(PYTHON) -m venv $(VENV)
	# Force upgrade to Torch 2.6+ for security and CUDA 12.1+ support
	$(PIP) install --upgrade "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || $(PIP) install --upgrade "torch>=2.6.0" torchvision torchaudio
	
	# Install AI dependencies
	$(PIP) install --upgrade diffusers "transformers==4.46.3" accelerate optimum-quanto bitsandbytes sentencepiece scipy protobuf imageio[ffmpeg] numpy torchsde

build_zpip:
	zig build-exe zpip.zig -O ReleaseSafe

build_rust:
	rustc assemble_video.rs -o assemble_video_rust

assets:
	# Run the generation script for Dalek
	$(PY_EXEC) generate_trailer_assets.py

video: build_rust
	# Assemble the Dalek video using Rust
	$(RUST_BIN) assets dalek_trailer.mpg dalek

snakes:
	# Run the generation script for Snakes on Titanic
	$(PY_EXEC) generate_snakes_assets.py

snakes_video: build_rust
	# Assemble the Snakes video using Rust
	$(RUST_BIN) assets_snakes snakes_on_titanic.mpg snakes

luftwaffe:
	# Run the generation script for Planet of the Luftwaffe
	$(PY_EXEC) generate_luftwaffe_assets.py

luftwaffe_video: build_rust
	# Assemble the Luftwaffe video using Rust
	$(RUST_BIN) assets_luftwaffe planet_of_the_luftwaffe.mpg luftwaffe

romcom:
	# Run the generation script for The Plunger's Refrain
	$(PY_EXEC) generate_romcom_assets.py

romcom_video: build_rust
	# Assemble the RomCom video using Rust
	$(RUST_BIN) assets_romcom plungers_refrain.mpg romcom

titanic2:
	# Run the generation script for Titanic 2: The Resinkening
	$(PY_EXEC) generate_titanic2_assets.py

titanic2_video: build_rust
	# Assemble the Titanic 2 video using Rust
	$(RUST_BIN) assets_titanic2 titanic2_resinkening.mpg titanic2

boring:
	# Run the generation script for The Damp Patch
	$(PY_EXEC) generate_boring_assets.py

boring_video: build_rust
	# Assemble the Boring video using Rust
	$(RUST_BIN) assets_boring the_damp_patch.mpg boring

wait:
	$(PY_EXEC) generate_wait_assets.py

assemble_wait: build_rust
	$(RUST_BIN) assets_wait everybody_wait.mpg wait

all_voiceovers:
	$(PY_EXEC) generate_trailer_assets.py voice
	$(PY_EXEC) generate_snakes_assets.py voice
	$(PY_EXEC) generate_luftwaffe_assets.py voice
	$(PY_EXEC) generate_romcom_assets.py voice
	$(PY_EXEC) generate_titanic2_assets.py voice
	$(PY_EXEC) generate_boring_assets.py voice
	$(PY_EXEC) generate_wait_assets.py voice

absurd:
	$(PY_EXEC) generate_absurd_trailers.py

assemble_absurd: build_rust
	for id in moistening gavelgeddon sentient_scone tax_audit_musical beige_alert stapler_heist lintpocalypse puddle_jumper cardboard_chronicles hover_grandma mildly_inconvenienced missing_sock 50_foot_toddler accountant_arthur elevator_pitch bean_man; do \
		$(RUST_BIN) assets_$$id absurd_$$id.mpg $$id; \
	done

wan:
	$(PY_EXEC) generate_wan_trailers.py

benchmark_tts:
	$(PY_EXEC) test_tts_benchmark.py

assemble_wan:
	$(PY_EXEC) generate_wan_trailers.py

all_trailers: video snakes_video luftwaffe_video romcom_video titanic2_video boring_video assemble_wait assemble_absurd assemble_wan

clean:
	rm -rf assets assets_snakes assets_luftwaffe assets_romcom assets_titanic2 assets_boring assets_wait assets_ltx
	rm -f *.mpg *.mp4
	rm -f assemble_video_rust assemble_video_cpp zpip
	rm -rf $(VENV)

.PHONY: all setup build_rust build_zpip assets video snakes snakes_video luftwaffe luftwaffe_video romcom romcom_video titanic2 titanic2_video boring boring_video wait assemble_wait all_trailers clean benchmark_tts

# --- ONNX Runtime / libtorch build settings ---
ONNXRUNTIME_DIR ?= /opt/homebrew/Cellar/onnxruntime/1.22.2_7
LIBTORCH_DIR ?= 

.PHONY: build-onnx build-cpp generate-assets-cpp

# Attempt to build and install onnx into the venv (best-effort)
build-onnx:
	bash build_onnx.sh

# Build the C++ asset generator linking against system ONNX Runtime
build-cpp:
	@echo "Building C++ asset generator using ONNX Runtime at $(ONNXRUNTIME_DIR)"
	clang++ -std=c++23 -O2 generate_trailer_assets_full.cpp -I$(ONNXRUNTIME_DIR)/include/onnxruntime -L$(ONNXRUNTIME_DIR)/lib -lonnxruntime -o generate_trailer_assets_full -Wl,-rpath,$(ONNXRUNTIME_DIR)/lib

# Run the generated C++ binary to produce assets
generate-assets-cpp: build-cpp
	./generate_trailer_assets_full

