import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# 페이지 설정 및 데이터 불러오기
# --------------------------------------------------
st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결 또는 데이터 주소를 확인해 주세요.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# 제목 및 입력 속성 선택
# --------------------------------------------------
st.title("🧠 분류 모델")
st.caption("뇌졸중(stroke=1)을 양성으로 두고 두 분류 모델의 성능과 판단 기준을 비교합니다.")

selected_features = st.multiselect(
    "모델 입력 속성을 선택하세요.",
    options=list(FEATURE_NAMES.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda col: FEATURE_NAMES[col],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# --------------------------------------------------
# 데이터 정렬 및 고정된 학습/테스트 분리
# 각 id 순서의 10명 중 앞 3명은 테스트, 나머지 7명은 학습
# --------------------------------------------------
data = df.sort_values("id").reset_index(drop=True).copy()
data["_순번"] = np.arange(len(data))
data["_묶음"] = data["_순번"] // 10
data["_묶음내순서"] = data["_순번"] % 10

test_mask = data["_묶음내순서"] < 3
train_df = data.loc[~test_mask].copy()
test_df = data.loc[test_mask].copy()

X_train_raw = train_df[selected_features].copy()
X_test_raw = test_df[selected_features].copy()
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# BMI를 선택한 경우에만 학습 데이터의 중앙값으로 결측치 대체
if "bmi" in selected_features:
    bmi_median = X_train_raw["bmi"].median()
    X_train_raw["bmi"] = X_train_raw["bmi"].fillna(bmi_median)
    X_test_raw["bmi"] = X_test_raw["bmi"].fillna(bmi_median)

# 훈련 데이터로만 크기 맞추기(표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train_raw, y_train)

# 기준 모델: 입력 속성 없이 학습 데이터의 다수 클래스로만 예측
majority_class = int(y_train.value_counts().idxmax())
baseline_train_pred = np.full(len(y_train), majority_class)
baseline_test_pred = np.full(len(y_test), majority_class)

# 모델별 정확도
log_train_acc = logistic_model.score(X_train_scaled, y_train)
log_test_acc = logistic_model.score(X_test_scaled, y_test)
tree_train_acc = tree_model.score(X_train_raw, y_train)
tree_test_acc = tree_model.score(X_test_raw, y_test)
base_train_acc = (baseline_train_pred == y_train.to_numpy()).mean()
base_test_acc = (baseline_test_pred == y_test.to_numpy()).mean()

# --------------------------------------------------
# 정확도 카드
# --------------------------------------------------
st.subheader("모델 정확도")
st.caption(
    f"전체 {len(data):,}명 중 테스트 {len(test_df):,}명, "
    f"훈련 {len(train_df):,}명으로 나누었습니다. "
    "각 10명 묶음의 앞 3명은 테스트용이며, 나머지 7명만 학습에 사용합니다."
)

card_cols = st.columns(3)

cards = [
    (
        "로지스틱 회귀(확률로 답하는 모델)",
        log_test_acc,
        log_train_acc,
        log_test_acc,
    ),
    (
        "의사결정트리(질문으로 답하는 모델)",
        tree_test_acc,
        tree_train_acc,
        tree_test_acc,
    ),
    (
        "다수 클래스 기준 모델",
        base_test_acc,
        base_train_acc,
        base_test_acc,
    ),
]

for col, (name, test_acc, train_acc, test_acc_again) in zip(card_cols, cards):
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.metric("테스트 데이터 정확도", f"{test_acc:.2%}")
            small1, small2 = st.columns(2)
            small1.caption(f"훈련 데이터: {train_acc:.2%}")
            small2.caption(f"테스트 데이터: {test_acc_again:.2%}")

st.divider()

# --------------------------------------------------
# 2개 속성 선택 및 분류 경계 시각화
# --------------------------------------------------
st.subheader("분류 경계 비교")
st.write("선택한 입력 속성 중 두 개를 골라 가로축과 세로축으로 사용합니다.")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda col: FEATURE_NAMES[col],
        key="x_feature",
    )

y_options = [col for col in selected_features if col != x_feature]
with axis_col2:
    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda col: FEATURE_NAMES[col],
        key="y_feature",
    )

# 두 축이 아닌 입력 속성은 테스트 데이터 중앙값에 고정
fixed_values = {}
for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test_raw[feature].median())

if fixed_values:
    fixed_text = ", ".join(
        f"{FEATURE_NAMES[col]} = {value:.2f}"
        for col, value in fixed_values.items()
    )
    st.write(f"**두 축 이외의 속성 고정값(테스트 데이터 중앙값):** {fixed_text}")
else:
    st.write("**두 축 이외의 속성:** 없습니다.")

# bmi 결측값은 이미 학습 중앙값으로 대체됨. 시각화 축 범위 생성
x_min = float(X_test_raw[x_feature].min())
x_max = float(X_test_raw[x_feature].max())
y_min = float(X_test_raw[y_feature].min())
y_max = float(X_test_raw[y_feature].max())

# 값이 모두 같을 때 축 범위 보정
if x_min == x_max:
    x_min -= 1
    x_max += 1
if y_min == y_max:
    y_min -= 1
    y_max += 1

x_pad = (x_max - x_min) * 0.08
y_pad = (y_max - y_min) * 0.08
x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 180)
y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 180)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_raw = pd.DataFrame({
    feature: np.full(xx.size, fixed_values.get(feature, 0.0))
    for feature in selected_features
})
grid_raw[x_feature] = xx.ravel()
grid_raw[y_feature] = yy.ravel()

# bmi 축일 경우 그리드에는 정상적인 값만 들어가며, 비축 BMI는 고정값에 결측 대체 적용
if "bmi" in selected_features:
    grid_raw["bmi"] = grid_raw["bmi"].fillna(
        float(X_train_raw["bmi"].median())
    )

grid_scaled = scaler.transform(grid_raw[selected_features])
log_grid_prob = logistic_model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
tree_grid_pred = tree_model.predict(grid_raw[selected_features]).reshape(xx.shape)

# 테스트 관측값 산점도: 실제 정답 레이블 색상
plot_df = X_test_raw[[x_feature, y_feature]].copy()
plot_df["실제 뇌졸중 여부"] = y_test.map({
    0: "뇌졸중 아님(0)",
    1: "뇌졸중(양성, 1)"
}).to_numpy()

fig_boundary = go.Figure()

# 트리 분류 영역을 먼저 옅게 표시
fig_boundary.add_trace(go.Contour(
    x=x_grid,
    y=y_grid,
    z=tree_grid_pred,
    showscale=False,
    hoverinfo="skip",
    contours=dict(
        start=0,
        end=1,
        size=1,
        coloring="fill",
        showlines=False
    ),
    opacity=0.16,
    name="의사결정트리 분류 영역",
    showlegend=True,
    colorscale=[
        [0.0, "rgba(70,130,220,0.30)"],
        [0.499, "rgba(70,130,220,0.30)"],
        [0.5, "rgba(230,90,90,0.30)"],
        [1.0, "rgba(230,90,90,0.30)"],
    ],
))

# 로지스틱 회귀 확률 0.5 경계
fig_boundary.add_trace(go.Contour(
    x=x_grid,
    y=y_grid,
    z=log_grid_prob,
    contours=dict(
        start=0.5,
        end=0.5,
        size=1,
        coloring="lines",
        showlabels=False,
    ),
    line=dict(width=3, color="#F4C542"),
    showscale=False,
    hoverinfo="skip",
    name="로지스틱 회귀 경계(확률 0.5)",
))

# 실제 테스트 데이터 점
for label, color in [
    ("뇌졸중 아님(0)", "#2E86DE"),
    ("뇌졸중(양성, 1)", "#E74C3C"),
]:
    subset = plot_df[plot_df["실제 뇌졸중 여부"] == label]
    fig_boundary.add_trace(go.Scatter(
        x=subset[x_feature],
        y=subset[y_feature],
        mode="markers",
        name=label,
        marker=dict(
            size=8,
            color=color,
            line=dict(width=0.7, color="white"),
            opacity=0.85,
        ),
        customdata=test_df.loc[subset.index, ["id"]].to_numpy(),
        hovertemplate=(
            f"{FEATURE_NAMES[x_feature]}: %{{x}}<br>"
            f"{FEATURE_NAMES[y_feature]}: %{{y}}<br>"
            "실제 분류: %{fullData.name}<br>"
            "ID: %{customdata[0]}<extra></extra>"
        ),
    ))

fig_boundary.update_layout(
    title="테스트 데이터와 두 모델의 분류 경계",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="범례",
    height=600,
)
st.plotly_chart(fig_boundary, use_container_width=True)

# 경계선이 보이는 영역에 존재하는지 점검
log_prob_min = float(np.min(log_grid_prob))
log_prob_max = float(np.max(log_grid_prob))
if not (log_prob_min <= 0.5 <= log_prob_max):
    st.info(
        "로지스틱 회귀의 확률 0.5 경계선은 현재 그림의 표시 범위 안에 없습니다."
    )

st.caption(
    "배경의 옅은 색은 의사결정트리의 예측 영역이며, 노란 선은 "
    "로지스틱 회귀가 양성 확률 0.5로 나누는 경계입니다. "
    "점의 색은 테스트 데이터의 실제 뇌졸중 여부를 나타냅니다."
)

st.divider()

# --------------------------------------------------
# 의사결정트리 질문 가지 그림 (Graphviz DOT 문자열)
# --------------------------------------------------
st.subheader("의사결정트리의 질문 구조")
st.caption(
    "각 마디에는 해당 훈련 데이터 수와 실제 뇌졸중(양성) 인원 및 비율을 표시합니다. "
    "예는 질문 조건을 만족하는 왼쪽 가지, 아니요는 오른쪽 가지입니다."
)

tree = tree_model.tree_
class_names = ["뇌졸중 아님(0)", "뇌졸중(양성, 1)"]

# tree_.value의 클래스별 개수는 학습 데이터 가중치가 없으므로 인원수로 해석 가능
dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="transparent"];',
    'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
    'edge [fontname="Arial", fontsize=10];',
]

leaf_nodes = []
used_features = set()

for node_id in range(tree.node_count):
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    is_leaf = left == right

    counts = tree.value[node_id][0]
    total = int(tree.n_node_samples[node_id])
    positive_index = int(np.where(tree_model.classes_ == 1)[0][0])
    class_total = float(np.sum(counts))
    positive_count = int(round((counts[positive_index] / class_total) * total)) if class_total else 0
    positive_rate = (positive_count / total * 100) if total else 0

    detail = (
        f"훈련용 {total}명\\n"
        f"뇌졸중 {positive_count}명 ({positive_rate:.1f}%)"
    )

    if is_leaf:
        predicted_class = int(tree_model.classes_[np.argmax(counts)])
        answer = "뇌졸중(양성)" if predicted_class == 1 else "뇌졸중 아님"
        fill = "#F8CACA" if predicted_class == 1 else "#CFE4FA"
        label = f"답: {answer}\\n{detail}"
        leaf_nodes.append((node_id, predicted_class))
    else:
        feature_index = tree.feature[node_id]
        feature_name = selected_features[feature_index]
        used_features.add(feature_name)
        threshold = tree.threshold[node_id]

        # BMI 외 변수는 모두 수치형. 조건을 우리말로 표시
        question = (
            f"{FEATURE_NAMES[feature_name]} ≤ {threshold:.2f} ?\\n"
            f"{detail}"
        )
        label = question
        fill = "#FFF2CC"

    # DOT label 줄바꿈 및 따옴표 안전 처리
    label = label.replace("\\", "\\\\").replace('"', '\\"')
    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="{fill}"];'
    )

for node_id in range(tree.node_count):
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    if left == right:
        continue

    dot_lines.append(f'{node_id} -> {left} [label="예"];')
    dot_lines.append(f'{node_id} -> {right} [label="아니요"];')

dot_lines.append("}")
dot_string = "\n".join(dot_lines)

st.graphviz_chart(dot_string, use_container_width=True)

leaf_count = len(leaf_nodes)
negative_leaf_count = sum(1 for _, cls in leaf_nodes if cls == 0)
used_feature_names = [FEATURE_NAMES[col] for col in selected_features if col in used_features]

st.write(f"**답을 내는 마디(잎)는 총 {leaf_count}칸이며, 그중 {negative_leaf_count}칸이 ‘뇌졸중 아님’으로 답합니다.**")
st.write(
    "**이 나무가 실제로 물은 속성:** "
    + (", ".join(used_feature_names) if used_feature_names else "없음")
)
