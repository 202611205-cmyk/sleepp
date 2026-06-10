import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="수면장애 예측 시스템", page_icon="🌙", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp { background-color: #0b0f1a; color: #e0e6f0; }
.hero { background: linear-gradient(135deg,#0d1b2e,#1a2d4a); border:1px solid #2d4a6a; border-radius:20px; padding:36px 40px; margin-bottom:28px; text-align:center; }
.hero h1 { font-size:30px; font-weight:700; color:#7ec8e3; margin:0; }
.hero p { font-size:14px; color:#7a93b8; margin-top:8px; }
.acc-badge { display:inline-block; background:#0a2235; border:1px solid #2d4a6a; border-radius:8px; padding:4px 12px; font-size:12px; color:#7ec8e3; margin-top:8px; }
.card { background:linear-gradient(135deg,#1a2235,#1e2d45); border:1px solid #2d4a6a; border-radius:16px; padding:24px; margin-bottom:16px; }
.card-title { font-size:15px; font-weight:600; color:#7ec8e3; border-left:3px solid #7ec8e3; padding-left:10px; margin-bottom:16px; }
.result-none { background:#0a2d1a; border:1px solid #2e7d32; border-radius:16px; padding:28px; text-align:center; }
.result-apnea { background:#3d1a1a; border:1px solid #c62828; border-radius:16px; padding:28px; text-align:center; }
.result-insomnia { background:#3d2f0a; border:1px solid #e65100; border-radius:16px; padding:28px; text-align:center; }
.result-label { font-size:26px; font-weight:700; margin-bottom:8px; }
.result-desc { font-size:13px; color:#a0b8d0; margin-top:10px; line-height:1.7; }
.tip-box { background:#0d1520; border-radius:10px; padding:12px 16px; margin-top:12px; font-size:12px; color:#7a93b8; line-height:2; }
.stSlider label, .stSelectbox label, .stNumberInput label { color:#a0b8d0 !important; font-size:13px !important; }
.stButton>button { background:linear-gradient(135deg,#1a4a6e,#2d6a9e); color:white; border:none; border-radius:12px; font-size:15px; font-weight:600; padding:12px; width:100%; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
    df[["BP_sys","BP_dia"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
    le_gender = LabelEncoder(); le_occ = LabelEncoder()
    le_bmi = LabelEncoder(); le_target = LabelEncoder()
    df["Gender_enc"] = le_gender.fit_transform(df["Gender"])
    df["Occ_enc"] = le_occ.fit_transform(df["Occupation"])
    df["BMI_enc"] = le_bmi.fit_transform(df["BMI Category"])
    df["Disorder_enc"] = le_target.fit_transform(df["Sleep Disorder"])
    features = ["Age","Gender_enc","Occ_enc","Sleep Duration","Quality of Sleep",
                "Physical Activity Level","Stress Level","BMI_enc","BP_sys","BP_dia","Heart Rate","Daily Steps"]
    X = df[features]; y = df["Disorder_enc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, le_gender, le_occ, le_bmi, le_target, features, acc, df

model, le_gender, le_occ, le_bmi, le_target, features, acc, df = load_model()
OCCUPATIONS = sorted(df["Occupation"].unique().tolist())
BMI_CATS = ["Normal","Normal Weight","Overweight","Obese"]
BMI_KR   = {"Normal":"정상","Normal Weight":"정상체중","Overweight":"과체중","Obese":"비만"}

st.markdown(f"""
<div class="hero">
    <h1>🌙 수면장애 예측 시스템</h1>
    <p>건강 정보를 입력하면 AI가 수면장애 유형과 위험도를 분석합니다</p>
    <span class="acc-badge">모델 정확도 {acc*100:.1f}% · Random Forest · 374명 실제 데이터</span>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown('<div class="card"><div class="card-title">👤 기본 정보</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gender     = st.selectbox("성별", ["Male","Female"], format_func=lambda x: "남성" if x=="Male" else "여성")
        age        = st.slider("나이", 20, 65, 35)
    with c2:
        occupation = st.selectbox("직업", OCCUPATIONS)
        bmi        = st.selectbox("BMI 상태", BMI_CATS, format_func=lambda x: BMI_KR[x])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">💤 수면 정보</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sleep_dur  = st.slider("수면시간 (시간)", 4.0, 9.0, 7.0, 0.1)
    with c2:
        sleep_qual = st.slider("수면의질 (1~10)", 1, 10, 7)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🏃 생활 습관</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        activity = st.slider("신체활동수준 (20~100)", 20, 100, 50)
        steps    = st.number_input("일일 걸음수", 1000, 15000, 7000, step=500)
    with c2:
        stress   = st.slider("스트레스지수 (1~10)", 1, 10, 5)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">❤️ 건강 수치</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        bp_sys     = st.number_input("혈압 수축기 (mmHg)", 90, 180, 120)
    with c2:
        bp_dia     = st.number_input("혈압 이완기 (mmHg)", 50, 120, 80)
    with c3:
        heart_rate = st.number_input("심박수 (bpm)", 50, 110, 70)
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍 수면장애 분석하기", use_container_width=True)

with col_right:
    if predict_btn:
        gender_enc = le_gender.transform([gender])[0]
        occ_enc    = le_occ.transform([occupation])[0]
        bmi_enc    = le_bmi.transform([bmi])[0]
        X_input    = pd.DataFrame([[age, gender_enc, occ_enc, sleep_dur, sleep_qual,
                                    activity, stress, bmi_enc, bp_sys, bp_dia, heart_rate, steps]],
                                  columns=features)
        pred_enc   = model.predict(X_input)[0]
        proba      = model.predict_proba(X_input)[0]
        pred_label = le_target.inverse_transform([pred_enc])[0]
        classes    = le_target.inverse_transform(model.classes_)

        cfg = {
            "None":        ("result-none",     "✅",  "수면장애 없음",     "#4ecdc4", "현재 건강 지표상 수면장애 위험이 낮습니다.<br>꾸준한 생활 습관을 유지하세요."),
            "Sleep Apnea": ("result-apnea",    "😮‍💨", "수면무호흡증 의심", "#ff6b6b", "수면 중 호흡이 반복적으로 멈출 수 있습니다.<br>이비인후과 또는 수면클리닉 방문을 권장합니다."),
            "Insomnia":    ("result-insomnia", "😶",  "불면증 의심",       "#ffd166", "잠들기 어렵거나 자주 깨는 증상이 나타날 수 있습니다.<br>수면 전문의 상담을 권장합니다."),
        }
        css_cls, icon, label, color, desc = cfg[pred_label]
        st.markdown(f"""
        <div class="{css_cls}">
            <div style="font-size:44px">{icon}</div>
            <div class="result-label" style="color:{color}">{label}</div>
            <div class="result-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">📊 유형별 예측 확률</div>', unsafe_allow_html=True)
        label_map = {"None":"정상","Sleep Apnea":"수면무호흡증","Insomnia":"불면증"}
        color_map = {"None":"#4ecdc4","Sleep Apnea":"#ff6b6b","Insomnia":"#ffd166"}
        for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
            c = color_map.get(cls,"#7ec8e3")
            st.markdown(f"""
            <div style="margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                    <span style="color:#a0b8d0;font-size:13px">{label_map.get(cls,cls)}</span>
                    <span style="color:{c};font-weight:700;font-size:13px">{prob*100:.1f}%</span>
                </div>
                <div style="background:#0d1520;border-radius:6px;height:10px">
                    <div style="width:{prob*100:.1f}%;background:{c};height:100%;border-radius:6px"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        importances = model.feature_importances_
        feat_kr = {"Age":"나이","Gender_enc":"성별","Occ_enc":"직업","Sleep Duration":"수면시간",
                   "Quality of Sleep":"수면의질","Physical Activity Level":"신체활동","Stress Level":"스트레스",
                   "BMI_enc":"BMI","BP_sys":"수축기혈압","BP_dia":"이완기혈압","Heart Rate":"심박수","Daily Steps":"걸음수"}
        top_idx   = np.argsort(importances)[::-1][:5]
        top_feats = [(feat_kr[features[i]], importances[i]) for i in top_idx]
        fig = go.Figure(go.Bar(
            x=[v for _,v in top_feats], y=[n for n,_ in top_feats], orientation="h",
            marker=dict(color=[v for _,v in top_feats], colorscale=[[0,"#1a4a6e"],[1,"#7ec8e3"]], showscale=False),
        ))
        fig.update_layout(
            title=dict(text="예측에 영향을 준 주요 요인", font=dict(color="#a0b8d0",size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a0b8d0",family="Noto Sans KR"),
            xaxis=dict(gridcolor="#2d3a52",showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=40,b=10), height=210,
        )
        st.plotly_chart(fig, use_container_width=True)

        tips = []
        if sleep_dur < 7:   tips.append("⏰ 수면시간을 7~8시간으로 늘려보세요")
        if stress >= 7:     tips.append("🧘 명상이나 호흡법으로 스트레스를 줄여보세요")
        if sleep_qual <= 5: tips.append("📵 잠들기 1시간 전 스마트폰 사용을 줄이세요")
        if steps < 5000:    tips.append("🚶 하루 7,000보 이상 걷기를 목표로 하세요")
        if bmi == "Obese":  tips.append("🥗 식이조절과 규칙적인 운동을 시작해보세요")
        if bp_sys >= 140:   tips.append("🩺 고혈압 관리를 위해 전문의 상담을 받으세요")
        if tips:
            st.markdown(f'<div class="tip-box">{"".join(f"<div>{t}</div>" for t in tips)}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:52px 24px">
            <div style="font-size:52px;margin-bottom:16px">🌙</div>
            <div style="font-size:18px;font-weight:600;color:#7ec8e3;margin-bottom:12px">건강 정보를 입력해주세요</div>
            <div style="font-size:13px;color:#7a93b8;line-height:2">
                왼쪽 폼에 성별, 나이, 수면 습관,<br>생활 패턴, 건강 수치를 입력하고<br>
                <strong style="color:#a0b8d0">분석 버튼</strong>을 누르면<br>AI가 수면장애 유형을 예측합니다.
            </div>
        </div>""", unsafe_allow_html=True)

        cnt = df["Sleep Disorder"].value_counts()
        label_map2 = {"None":"정상","Sleep Apnea":"수면무호흡증","Insomnia":"불면증"}
        color_map2 = {"None":"#4ecdc4","Sleep Apnea":"#ff6b6b","Insomnia":"#ffd166"}
        st.markdown('<div class="card"><div class="card-title">📈 학습 데이터 현황</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=[label_map2.get(k,k) for k in cnt.index], values=cnt.values, hole=0.55,
            marker_colors=[color_map2.get(k,"#7ec8e3") for k in cnt.index],
            textfont=dict(color="white",size=12),
        ))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a0b8d0",family="Noto Sans KR"),
                           legend=dict(font=dict(color="#a0b8d0"),bgcolor="rgba(0,0,0,0)"),
                           margin=dict(l=0,r=0,t=0,b=0), height=200)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f'<div style="text-align:center;color:#5a7a9a;font-size:12px">전체 {len(df)}명 데이터 기반 Random Forest 모델</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:20px;color:#2d4a6a;font-size:11px;margin-top:20px;border-top:1px solid #1e2d3d">🌙 이 예측 결과는 의학적 진단을 대체하지 않습니다. 이상 증상이 있으면 전문의와 상담하세요.</div>', unsafe_allow_html=True)
