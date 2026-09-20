import streamlit as st
import pandas as pd

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

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
# 제목 및 소개
# --------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.markdown(
    """
    뇌졸중 예측 실습에 사용하는 데이터를 살펴보는 공간입니다.
    데이터의 기본 구조와 변수별 정보를 확인하고, 이후 분석 및 예측 실습에 활용합니다.
    """
)

st.divider()

# --------------------------------------------------
# 핵심 데이터 카드
# --------------------------------------------------
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_rate = (stroke_count / total_people * 100) if total_people else 0

st.subheader("데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 사람 수", f"{total_people:,}명")
col2.metric("열 개수", f"{column_count:,}개")
col3.metric("뇌졸중 경험자 수", f"{stroke_count:,}명")
col4.metric("뇌졸중 경험 비율", f"{stroke_rate:.2f}%")

st.divider()

# --------------------------------------------------
# 열 정보 표
# --------------------------------------------------
st.subheader("데이터 열 정보")
st.caption("우리말 뜻은 교재를 참고해 직접 작성할 수 있도록 비워 두었습니다.")

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": ["" for _ in df.columns],
    "값의 종류": [df[col].nunique(dropna=True) for col in df.columns],
    "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns]
})

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# 데이터 미리보기
# --------------------------------------------------
st.subheader("데이터 미리보기")
st.caption("원본 데이터의 처음 다섯 행입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# 데이터 출처
# --------------------------------------------------
st.subheader("데이터 출처")
st.write("출처: ______________________________________________")
