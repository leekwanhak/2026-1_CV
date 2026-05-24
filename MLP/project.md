# PROJECT: MLP 신경망 성능 영향 요소 분석 (HW#3)

> 이 문서는 Claude가 본 프로젝트의 컨텍스트를 정확히 이해하기 위한 명세 파일이다.
> 모든 작업(코드 작성, 시각화, 보고서 분석)은 이 문서의 요구사항을 충족해야 한다.

---

## 1. 프로젝트 개요

### 1.1 목표
손실 함수(Loss Function), 활성화 함수(Activation Function), 최적화 알고리즘(Optimizer)이 MLP 학습 결과에 미치는 영향을 **구현과 실험을 통해 정량적으로 분석**하고, 결과 기반 비교 분석 보고서를 작성한다.

### 1.2 마감 및 제출물
- **마감:** 2026년 5월 24일(일) 23:59
- **제출물:**
  1. 보고서 (PDF 형식)
  2. 실험 코드 ipynb로 작성 — **GitHub 업로드 후 링크 제출**
  3. 코드 주석 필수 (구체적으로)

---

## 2. 기본 네트워크 구조

```python
# 예시 - 변경 가능 (Deeper/Shallower도 추가 실험하면 좋음)
model = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)
```

추가 실험 후보:
- **Deeper:** `512 → 256 → 128 → Output`
- **Shallow:** `128 → 64 → Output`

---

## 3. 구현 제약 (반드시 준수)

### 3.1 프레임워크
- PyTorch 또는 TensorFlow 사용
- 모델은 `nn.Module` 또는 `tf.keras.Model`로 구성 가능
- **학습 루프는 `for epoch in range(...)` 형태로 직접 작성 (필수)**

### 3.2 자동 미분
- `backward()` (PyTorch) 또는 `tape.gradient()` (TF) 사용 가능
- **단, 손실 함수와 활성화 함수는 명시적으로 설정·변경할 것**

### 3.3 명시적 구성
- `loss_fn = ...` 형식으로 명확히 설정
- `optimizer = torch.optim.Adam(...)` 등 명확히 기술
- 실험별로 이 구성 요소를 바꿔가며 비교

### 3.4 평가 모드
- 평가 시 **`model.eval()` + `with torch.no_grad()` 필수 사용**

### 3.5 학습률 가이드
| Optimizer | 권장 lr 범위 | 이유 |
|---|---|---|
| SGD | 0.01 ~ 0.1 | 단순 경사하강법, 비교적 큰 lr 사용 |
| SGD+Momentum | 0.01 ~ 0.05 | 진동을 줄이기 위해 약간 낮춤 |
| Adam | 0.001 ~ 0.01 | Adaptive Learning Rate 적용 |

### 3.6 에폭 수 가이드
| 데이터셋 | 권장 epoch |
|---|---|
| Fashion-MNIST | 20 ~ 50 |
| Digits Dataset | 20 ~ 50 |
| make_moons | 200 ~ 500 |
| CIFAR-10 | 50 ~ 100 |

- Baseline epoch 먼저 설정
- 수렴 속도가 느리면 **Learning Rate Decay** 적용

---

## 4. 실험 A — 손실 함수 비교

### 4.1 비교 대상
**CrossEntropy Loss vs MSE Loss (with softmax)**

### 4.2 목표
- MSE와 CrossEntropy가 학습 성능에 미치는 차이 분석
- 학습 곡선의 수렴 속도, 정확도, loss 안정성 비교

### 4.3 데이터셋
- Fashion-MNIST
- Digits Dataset (scikit-learn, 8x8)

### 4.4 실험 조건
- 동일한 네트워크 구조 및 optimizer
- **손실 함수만 변경**
- MSE 사용 시 출력층에 **softmax 명시적 적용** (CrossEntropy는 내부적으로 softmax 포함)

```python
# MSE 사용 시
loss_fn = nn.MSELoss()
outputs = torch.softmax(model(inputs), dim=1)
```

### 4.5 분석 포인트
- MSE 사용 시 Gradient Vanishing 문제
- CrossEntropy의 빠른 수렴 속도와 안정성
- 최종 정확도 차이, 학습 곡선 비교

### 4.6 분석 질문 (보고서에 반드시 답변)
1. 왜 MSE는 CrossEntropy보다 학습이 느리고 안정적이지 못한가? (logit 분포 변화 + gradient scaling 관점)
2. CrossEntropy가 MSE보다 수렴 속도가 빠른 이유? (Gradient Vanishing 완화 관점)
3. 학습 초기 vs 후반의 gradient 분포 차이는?
4. MSE 사용 시 softmax를 적용하지 않으면 학습이 왜 어려운가?

---

## 5. 실험 B — 활성화 함수 비교

### 5.1 비교 대상
**ReLU vs LeakyReLU vs Sigmoid**

### 5.2 목표
- 세 활성화 함수가 학습에 미치는 영향 분석
- **Dead ReLU 발생 유도** 및 LeakyReLU의 완화 효과 확인

### 5.3 데이터셋
- make_moons
- make_circles
(2D 비선형 분류 — Dead ReLU 및 학습 경계 시각화에 유리)

### 5.4 실험 조건
- 동일 네트워크 + 손실 함수(CrossEntropy) + Optimizer (Adam 또는 SGD)
- **weight 초기값을 `std=0.01` 등 작게 설정**하여 dead neuron 상황 유도

### 5.5 분석 포인트
- Layer별 출력값 시각화 (matplotlib)
- Sigmoid → vanishing gradient 발생 → 학습 정체 확인
- Dead ReLU 발생 시 어떤 지표가 정체되는지 분석
- **Dead ReLU 발생 neuron 비율 측정 → 히트맵 시각화**
- LeakyReLU의 Dead ReLU 완화 효과 시각적 분석

### 5.6 분석 질문 (보고서에 반드시 답변)
1. Dead ReLU 발생 비율이 높으면 학습에 미치는 영향은? (특정 Layer 정체 원인)
2. LeakyReLU가 Dead ReLU를 얼마나 완화하는가?
3. Sigmoid 사용 시 vanishing gradient가 발생하는 구간은?
4. 각 Layer가 학습에 기여하는 정도를 히트맵으로 분석

---

## 6. 실험 C — 최적화 알고리즘 비교

### 6.1 비교 대상
**SGD vs SGD+Momentum vs Adam**

### 6.2 목표
- 세 optimizer의 성능 비교
- 학습률 변화 영향 분석 (**0.1, 0.01, 0.001** 세 가지)

### 6.3 데이터셋
- Fashion-MNIST
- Digits Dataset

### 6.4 실험 조건
- 동일 네트워크, 손실 함수, 데이터셋
- 학습률 × optimizer 조합 성능 비교 (**표로 정리**)
- **Exponential Decay 적용 실험 포함**

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
```

### 6.5 분석 포인트
- Overshooting, 느린 수렴, 안정성
- 학습률 변화에 따른 Loss 곡선과 Gradient 흐름

### 6.6 분석 질문 (보고서에 반드시 답변)
1. SGD/SGD+Momentum/Adam의 학습 곡선이 다른 이유? (Momentum이 SGD보다 빠른 수렴 이유)
2. 학습률 변화가 학습 안정성에 미치는 영향? (overshooting vs 정체)
3. Adam이 초기 학습에 빠르게 수렴하는 이유? (Adaptive LR 관점)
4. Exponential Decay 적용 시 성능 변화? (과적합 감소 / 수렴 안정성)
5. Gradient 흐름이 특정 Optimizer에서 사라지거나 급격히 변할 때 발생 문제? (Vanishing/Exploding)
6. 동일 네트워크에서 Optimizer마다 다른 학습 패턴을 보이는 이유? **(수식적 근거 포함)**

---

## 7. 공통 분석 질문 (모든 실험 적용)

1. 손실 함수·활성화 함수·Optimizer가 학습 곡선에 미치는 영향? (수렴 속도, 진동, 학습 정체 구간 그래프로 설명)
2. Loss 감소와 Accuracy 상승 간 불균형 발생 시 원인과 개선 방안? (특정 시점에서 Loss는 감소하지만 Accuracy가 오르지 않을 때 원인을 분석)
3. Layer별 Activation 분포가 학습 중 어떻게 변화하는가? (초반, 중반, 후반 학습 시 활성화 함수 (ReLU, Sigmoid) 값의 분포를 시각화하고 분석)
4. Gradient Flow가 특정 층에서 소멸/폭발할 때 모델에 미치는 영향? (Gradient Vanishing/Exploding이 발생한 구간을 찾아 원인을 분석)

---

## 8. 시각화 요구사항

### 8.1 그래프 (필수)
- **Loss vs Epoch** 곡선
- **Accuracy vs Epoch** 곡선
- 활성화 함수 실험 시: **중간 레이어 출력 분포** (히스토그램 또는 BoxPlot)

### 8.2 히트맵 (필수)
- **Dead ReLU Neuron 비율** 시각화 — 각 레이어의 뉴런 활성화 값을 히트맵으로
- **Gradient 흐름** 시각화 — 어느 Layer에서 gradient가 소멸하는지 시각적으로 명확히

---

## 9. 정량 비교 표 (필수)

### 9.1 실험 A 표
| 실험 항목 | 최종 정확도 (%) | Loss 최솟값 | 수렴 epoch |
|---|---|---|---|
| CrossEntropy | | | |
| MSE (with softmax) | | | |

### 9.2 실험 B 표
| 활성화 함수 | Dead ReLU 비율 (%) | 최종 정확도 (%) | 수렴 속도 |
|---|---|---|---|
| ReLU | | | |
| LeakyReLU | | | |
| Sigmoid | | | |

### 9.3 실험 C 표
| Optimizer | 최종 정확도 (%) | 수렴 속도 | 안정성 |
|---|---|---|---|
| SGD | | | |
| SGD+Momentum | | | |
| Adam | | | |

---

## 10. 보고서 구조 (각 실험별)

1. 실험 목표
2. 코드 및 실행 결과 — **코드 안에 주석으로 처리, 보고서에 따로 넣지 않아도 됨**
3. 그래프 및 시각화 결과
4. 정량적 분석 (표 포함)
5. 해설 및 분석 (질문 답변)
   - **단순 "정확도가 더 높았다" 수준 금지**
   - **원인/기전 중심 서술 필수**
6. 결론 및 개선 사항

---

## 11. 평가 기준 (각 20%)

### 11.1 실험 정확성 및 재현성 (20%)
- 동일 조건 재현 가능한가?
- 학습률, 에폭 수, Optimizer 설정 명확히 기록했는가?

### 11.2 학습곡선 분석 능력 (20%)
- Loss/Accuracy 진동 구간 해석
- 학습 빠른 진행 시점 vs 정체 시점 시각적 명확화

### 11.3 Activation 분포 분석 (20%)
- Dead ReLU 발생 Layer를 히트맵으로 정확히 시각화
- Sigmoid의 Vanishing Gradient 구간 설명
- LeakyReLU의 Dead ReLU 완화 효과 명확한 분석

### 11.4 Optimizer 비교 분석 (20%)
- 세 optimizer의 수렴 속도 차이 시각화
- Local Minima 회피를 위한 최적화 기법 차이 설명
- Exponential Decay의 학습 안정성 영향 분석

### 11.5 문제 해결 접근법 (20%)
- 진동/수렴 어려움 발생 시 해결 방안 제시
- Dead ReLU 심각 시 해결 방안 제시

---

## 12. 작업 체크리스트

### 12.1 공통 인프라
- [x] 데이터로더 모듈 (Fashion-MNIST, Digits, make_moons, make_circles)
- [x] 학습 루프 모듈 (`for epoch in range(...)` 직접 작성)
- [x] 평가 함수 (`model.eval()` + `torch.no_grad()`)
- [x] 시각화 유틸 (Loss/Acc 곡선, 히스토그램, 히트맵, gradient flow)
- [x] Random seed 고정 (재현성)

### 12.2 실험 A
- [x] CrossEntropy 실험 (Fashion-MNIST + Digits)
- [x] MSE + softmax 실험 (Fashion-MNIST + Digits)
- [x] Loss/Acc 곡선, Gradient 분포 그래프
- [x] 정량 비교 표
- [ ] 분석 질문 4개 답변

### 12.3 실험 B
- [x] ReLU 실험 (weight std=0.01)
- [x] LeakyReLU 실험
- [x] Sigmoid 실험
- [x] Loss/Acc 곡선
- [x] Activation 분포 히스토그램/BoxPlot
- [x] Dead ReLU 비율 히트맵
- [x] Gradient flow 히트맵
- [x] 정량 비교 표
- [ ] 분석 질문 4개 답변

### 12.4 실험 C
- [x] SGD × {0.1, 0.01, 0.001}
- [x] SGD+Momentum × {0.1, 0.01, 0.001}
- [x] Adam × {0.1, 0.01, 0.001}
- [x] ExponentialLR 적용 vs 미적용 비교
- [x] Loss/Acc 곡선 (optimizer × lr 그리드)
- [x] Gradient norm 시각화
- [x] 정량 비교 표 (그리드 표 + 종합 표 + Decay 비교 표)
- [ ] 분석 질문 6개 답변 (수식 포함)

### 12.5 보고서 작성
- [ ] 공통 질문 4개 답변
- [ ] 각 실험 6개 항목 작성
- [ ] 종합 결론
- [ ] PDF 변환

### 12.6 코드 제출
- [x] 주석 충실히
- [ ] README 작성 (실행 방법)
- [ ] GitHub 업로드
- [ ] 링크 제출

---

## 13. 작업 시 주의사항

- **분석은 원인/기전 중심으로 서술** ("정확도가 더 높았다" 수준 금지)
- 수식적 근거가 필요한 곳은 LaTeX 수식 포함 (특히 실험 C Q6)
- 모든 실험은 **동일 조건에서 재현 가능**해야 함 (seed 고정, 설정 기록)
- 코드 주석은 구체적으로 — 어떤 라인이 어떤 분석 포인트에 대응되는지 명시
- 시각화는 단순 출력이 아니라 **분석에 활용**되어야 함 (그래프마다 "무엇을 보이는가" 명확히)
