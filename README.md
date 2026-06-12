
---

## 1. 환경 요구사항

- Python 3.9 이상
- Git, Git LFS

---

## 2. 레포지토리 클론

```bash
# Git LFS 초기화 (최초 1회)
git lfs install

# 레포 클론
git clone https://github.com/Kangsuyeon01/standard_tactile_demo.git
cd standard_tactile_demo
```

---

## 3. Python 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# ONNX Runtime 추가 설치 (실시간 추론 최적화용)
pip install onnx>=1.14 onnxruntime>=1.16
```

---

## 4. 실행 방법

### 소켓 통신 실행 (Unity 연동용)

```bash
python -m scripts.realtime --pt-path pt_files/runs/20260612-003/best_model.pt --cache-path pt_files/inference_cache_allinone.npz
```

