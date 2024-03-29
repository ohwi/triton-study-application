# 테스트 실행

```bash
docker build . -t ohwi/triton_test && docker run --rm --gpus  '"device=0"' ohwi/triton_test
```

# 테스트 결과

테스트는 cuda 12.2 버전이 설치된 A100 40GB에서 진행됐습니다.

테스트는 `TransfomerEngine`에서 `FusedRoPE`와 unfused version을 [테스트 하기 위해서 사용한 함수](https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/test_fused_rope.py)에서 디폴트를 `seq_length: 4096`, `hidden_size: 128`, `rotary_percent: 1.0`, `batch_size: 16`, `head_num: 32`, `margin: 0`로 설정하고 하나씩 숫자를 바꿔가면서 진행 했습니다.
비교를 위해 `TransfomerEngine`에 구현되어 있는 `Torch native` unfused version, fused version, triton 구현체를 비교했습니다.

## Version 1

### `seq_length` = 512, 1024, 2048, 4096
```
RoPE-performance-version-1-seq_length-test:
   seq_length  Torch Native  Transformer Engine (Fused)    Triton
0       512.0      2.506752                    0.711680  0.713728
1      1024.0      4.720640                    1.360896  1.373184
2      2048.0      9.277440                    2.646016  2.646016
3      4096.0     18.423807                    5.210112  5.197824
```

![RoPE-performance-version-1-seq_length-test.png](artifacts%2Fversion-1%2FRoPE-performance-version-1-seq_length-test.png)



### `hidden_size` = 128, 256, 512

```
RoPE-performance-version-1-hidden_size-test:
   hidden_size  Torch Native  Transformer Engine (Fused)     Triton
0        128.0     18.426880                    5.208064   5.197824
1        256.0     36.747265                   11.131392   9.598976
2        512.0     73.428986                   24.171520  18.699265
```

![RoPE-performance-version-1-hidden_size-test.png](artifacts%2Fversion-1%2FRoPE-performance-version-1-hidden_size-test.png)

`hidden_size`가 커지면 속도 차이가 생기는 점에서 쓰레드 블럭에 관련된 파라미터를 조절할 필요가 보였습니다. V1에서는 1d 쓰레드 블럭을 사용했는데, 전체 head에서 사용되는 연산이 같은 점을 이용해서 2d 쓰레드 블럭으로 확장이 가능해보였습니다.



### `rotary_percent` = 0.5, 1.0

```
RoPE-performance-version-1-rotary_percent-test:
   rotary_percent  Torch Native  Transformer Engine (Fused)    Triton
0             0.5      14.11584                    5.099520  7.843840
1             1.0      18.42688                    5.211136  5.197824
```

![RoPE-performance-version-1-rotary_percent-test.png](artifacts%2Fversion-1%2FRoPE-performance-version-1-rotary_percent-test.png)

`rotary_percent`가 작으면 속도 많이 느려졌습니다. `rotary_percent`가 작으면 `hidden_size`가 작아지는 것과 비슷한 현상이 보이게 될 것이고 같은 문제에서 오는 현상으로 판단했습니다.



### `batch_size` = 2, 4, 8, 16

```
RoPE-performance-version-1-batch_size-test:
   batch_size  Torch Native  Transformer Engine (Fused)    Triton
0         2.0      2.425856                    0.726016  0.720896
1         4.0      4.721664                    1.375232  1.376256
2         8.0      9.276416                    2.648064  2.649088
3        16.0     18.429953                    5.208064  5.197824
```

![RoPE-performance-version-1-batch_size-test.png](artifacts%2Fversion-1%2FRoPE-performance-version-1-batch_size-test.png)




### `head_num` = 8, 16, 32, 64

```
RoPE-performance-version-1-head_num-test:
   head_num  Torch Native  Transformer Engine (Fused)     Triton
0       8.0      4.721664                    1.444864   1.408000
1      16.0      9.275392                    2.828288   2.671616
2      32.0     18.427904                    5.208064   5.197824
3      64.0     36.743683                   10.625024  10.288640
```

![RoPE-performance-version-1-head_num-test.png](artifacts%2Fversion-1%2FRoPE-performance-version-1-head_num-test.png)




### `margin` = 0, 10, 33, 77

```
RoPE-performance-version-1-margin-test:
   margin  Torch Native  Transformer Engine (Fused)    Triton
0     0.0     18.425856                    5.208064  5.198848
1    10.0     18.384895                    5.198848  5.188608
2    33.0     18.279425                    5.171200  5.158912
3    77.0     18.083839                    5.113856  5.105664
```

![RoPE-performance-version-1-margin-test.png](artifacts%2Fversion-1%2FRoPE-performance-version-1-margin-test.png)


## Version 2

V1에서 문제점이라고 판단한 부분을 수정했습니다.

1. 1d 쓰레드 블럭 대신 2d 쓰레드 블럭을 사용
2. `head_num` x `hidden_size`를 기준으로 warp 숫자와 block size를 조절

결과:

### `seq_length` = 512, 1024, 2048, 4096

```
RoPE-performance-version-2-seq_length-test:
   seq_length  Torch Native  Transformer Engine (Fused)    Triton
0       512.0      2.423808                    0.711680  0.670720
1      1024.0      4.721664                    1.367040  1.295360
2      2048.0      9.278464                    2.647040  2.512896
3      4096.0     18.424831                    5.233664  4.964352
```

![RoPE-performance-version-2-seq_length-test.png](artifacts%2Fversion-2%2FRoPE-performance-version-2-seq_length-test.png)


### `hidden_size` = 128, 256, 512

```
RoPE-performance-version-2-hidden_size-test:
   hidden_size  Torch Native  Transformer Engine (Fused)     Triton
0        128.0     18.429953                    5.233664   4.964352
1        256.0     36.751873                   11.133440  10.204160
2        512.0     73.436157                   24.182783  22.141441
```

![RoPE-performance-version-2-hidden_size-test.png](artifacts%2Fversion-2%2FRoPE-performance-version-2-hidden_size-test.png)



### `rotary_percent` = 0.5, 1.0

```
RoPE-performance-version-2-rotary_percent-test:
   rotary_percent  Torch Native  Transformer Engine (Fused)    Triton
0             0.5     14.116864                     5.09952  4.785152
1             1.0     18.429953                     5.23264  4.963328
```

![RoPE-performance-version-2-rotary_percent-test.png](artifacts%2Fversion-2%2FRoPE-performance-version-2-rotary_percent-test.png)


### `batch_size` = 2, 4, 8, 16

```
RoPE-performance-version-2-batch_size-test:
   batch_size  Torch Native  Transformer Engine (Fused)    Triton
0         2.0      2.427904                    0.722944  0.667648
1         4.0      4.722688                    1.378304  1.285120
2         8.0      9.279488                    2.649088  2.493440
3        16.0     18.426880                    5.233664  4.963328
```

![RoPE-performance-version-2-batch_size-test.png](artifacts%2Fversion-2%2FRoPE-performance-version-2-batch_size-test.png)



### `head_num` = 8, 16, 32, 64

```
RoPE-performance-version-2-head_num-test:
   head_num  Torch Native  Transformer Engine (Fused)     Triton
0       8.0      4.723712                    1.445888   1.314816
1      16.0      9.278464                    2.829312   2.468864
2      32.0     18.428928                    5.233664   4.964352
3      64.0     36.738045                   10.626048  10.165248
```

![RoPE-performance-version-2-head_num-test.png](artifacts%2Fversion-2%2FRoPE-performance-version-2-head_num-test.png)


### `margin` = 0, 10, 33, 77

```
RoPE-performance-version-2-margin-test:
   margin  Torch Native  Transformer Engine (Fused)    Triton
0     0.0     18.428928                    5.234688  4.964352
1    10.0     18.385920                    5.223424  4.954624
2    33.0     18.278400                    5.193728  4.927488
3    77.0     18.087936                    5.139456  4.875264
```

![RoPE-performance-version-2-margin-test.png](artifacts%2Fversion-2%2FRoPE-performance-version-2-margin-test.png)


V1에 비해서 `hidden_size`가 512인 경우의 속도가 조금 느려졌습니다. 하지만 `rotary_percent`가 0.5인 경우의 문제점을 개선하고, `TransformerEngine`의 `FusedRoPE`와 비슷한 속도의 triton 버전을 만들 수 있었습니다.


# 결과 값이 같다를 보여주는 테스트 결과

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-7.4.0, pluggy-1.2.0
rootdir: /workspace
plugins: shard-0.1.2, xdist-3.3.1, flakefinder-1.1.0, rerunfailures-12.0, xdoctest-1.0.2, hypothesis-5.35.1
collected 576 items
Running 576 items in this shard

compare_rope.py ........................................................ [  9%]
........................................................................ [ 22%]
........................................................................ [ 34%]
........................................................................ [ 47%]
........................................................................ [ 59%]
........................................................................ [ 72%]
........................................................................ [ 84%]
........................................................................ [ 97%]
................                                                         [100%]

=============================== warnings summary ===============================
../usr/local/lib/python3.10/dist-packages/setuptools/__init__.py:9
  /usr/local/lib/python3.10/dist-packages/setuptools/__init__.py:9: DeprecationWarning: The distutils package is deprecated and slated for removal in Python 3.12. Use setuptools or check PEP 632 for potential alternatives
    import distutils.core

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 576 passed, 1 warning in 38.31s ========================
```
