import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# 페이지 설정 및 데이터 불러오기
# --------------------------------------------------
st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결 또는 데이터 주소를 확인해 주세요."
    )
    st.exception(e)
    st.stop()

# 분석용 복사본
data = df.copy()

# --------------------------------------------------
# 페이지 제목
# --------------------------------------------------
st.title("🔎 데이터 탐색")
st.caption(
    "뇌졸중 여부에 따른 주요 변수의 분포와 비율을 살펴봅니다."
)

st.divider()

# --------------------------------------------------
# 1. 나이 및 평균 혈당 분포
# --------------------------------------------------
st.subheader("1. 나이와 평균 혈당의 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age = px.histogram(
        data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    fig_age.update_layout(
        bargap=0.08,
        yaxis_title="사람 수"
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )

with hist_col2:
    fig_glucose = px.histogram(
        data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    fig_glucose.update_layout(
        bargap=0.08,
        yaxis_title="사람 수"
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )

st.divider()

# --------------------------------------------------
# 2. 뇌졸중 여부별 나이 및 평균 혈당 상자그림
# --------------------------------------------------
st.subheader(
    "2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교"
)

data["뇌졸중 여부"] = data["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 나이",
        labels={
            "age": "나이",
            "뇌졸중 여부": "뇌졸중 여부"
        },
        category_orders={
            "뇌졸중 여부": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        }
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )

with box_col2:
    fig_glucose_box = px.box(
        data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 평균 혈당",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "뇌졸중 여부"
        },
        category_orders={
            "뇌졸중 여부": [
                "뇌졸중 없음",
                "뇌졸중 있음"
            ]
        }
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )

# 두 그룹의 평균값
means = (
    data.groupby("뇌졸중 여부")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex([
        "뇌졸중 없음",
        "뇌졸중 있음"
    ])
    .rename(columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    })
    .reset_index()
)

means[["평균 나이", "평균 혈당"]] = (
    means[["평균 나이", "평균 혈당"]].round(2)
)

st.markdown("**뇌졸중 여부별 평균값**")

st.dataframe(
    means,
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# 3. 고혈압 및 심장병 여부별 뇌졸중 비율
# --------------------------------------------------
st.subheader(
    "3. 고혈압·심장병 여부에 따른 뇌졸중 비율"
)


def stroke_rate_by_condition(column, label):
    summary = (
        data.groupby(column, dropna=False)["stroke"]
        .agg(
            전체_인원="count",
            뇌졸중_인원="sum",
            뇌졸중_비율="mean"
        )
        .reset_index()
    )

    summary["뇌졸중_비율(%)"] = (
        summary["뇌졸중_비율"] * 100
    )

    summary[label] = (
        summary[column]
        .map({
            0: "없음",
            1: "있음"
        })
        .fillna("알 수 없음")
    )

    return summary


rate_col1, rate_col2 = st.columns(2)

# 고혈압 여부
with rate_col1:
    hypertension_summary = stroke_rate_by_condition(
        "hypertension",
        "고혈압"
    )

    fig_hypertension = px.bar(
        hypertension_summary,
        x="고혈압",
        y="뇌졸중_비율(%)",
        text=hypertension_summary[
            "뇌졸중_비율(%)"
        ].map(lambda x: f"{x:.2f}%"),
        title="고혈압 여부별 뇌졸중 비율",
        labels={
            "고혈압": "고혈압",
            "뇌졸중_비율(%)": "뇌졸중 비율(%)"
        }
    )

    fig_hypertension.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True
    )

# 심장병 여부
with rate_col2:
    heart_summary = stroke_rate_by_condition(
        "heart_disease",
        "심장병"
    )

    fig_heart = px.bar(
        heart_summary,
        x="심장병",
        y="뇌졸중_비율(%)",
        text=heart_summary[
            "뇌졸중_비율(%)"
        ].map(lambda x: f"{x:.2f}%"),
        title="심장병 여부별 뇌졸중 비율",
        labels={
            "심장병": "심장병",
            "뇌졸중_비율(%)": "뇌졸중 비율(%)"
        }
    )

    fig_heart.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True
    )

st.divider()

# --------------------------------------------------
# 4. BMI 결측자와 전체 뇌졸중 비율 비교
# --------------------------------------------------
st.subheader(
    "4. BMI 빈 값이 있는 사람들의 뇌졸중 비율"
)

missing_bmi = data[data["bmi"].isna()]

overall_rate = data["stroke"].mean() * 100

missing_bmi_rate = (
    missing_bmi["stroke"].mean() * 100
    if len(missing_bmi) > 0
    else 0
)

bmi_comparison = pd.DataFrame({
    "대상": [
        "BMI 빈 값이 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        len(missing_bmi),
        len(data)
    ],
    "뇌졸중 경험자 수": [
        int(missing_bmi["stroke"].sum()),
        int(data["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_rate, 2),
        round(overall_rate, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# 5. 흡연 상태별 사람 수
# --------------------------------------------------
st.subheader("5. 흡연 상태별 사람 수")

smoking_counts = (
    data["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_counts["흡연 상태"] = (
    smoking_counts["흡연 상태"].fillna("빈 값")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True
)
