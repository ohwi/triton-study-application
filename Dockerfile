FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN MAKEFLAGS="-j32" pip install flash-attn==2.4.2 --no-build-isolation
RUN git clone --recursive https://github.com/NVIDIA/TransformerEngine.git && cd TransformerEngine && git checkout b8eea8aaa94bb566c3a12384eda064bda8ac4fd7 && export NVTE_FRAMEWORK=pytorch && MAKEFLAGS="-j32" pip install -e .

COPY ./rope.py /workspace/rope.py
COPY ./rope_triton.py /workspace/rope_triton.py
COPY ./rope_triton_v2.py /workspace/rope_triton_v2.py
COPY ./compare_rope.py /workspace/compare_rope.py
COPY ./benchmark.py /workspace/benchmark.py
COPY ./benchmark_v2.py /workspace/benchmark_v2.py

COPY ./run_tests.sh /workspace/run_tests.sh

ENTRYPOINT ["bash", "/workspace/run_tests.sh"]
