# Run with:
# python -m streamlit run app.py

import base64
import html
import sqlite3
from contextlib import closing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel


# ---------------------------------------------------------------- data
BASE = Path(__file__).resolve().parent

ART = BASE / "artifacts"
DS = BASE / "datasets"
ASSETS = BASE / "assets"
DATA = BASE / "data"

IMG = "https://image.tmdb.org/t/p/w500"

N_RECS, POOL, ALPHA = 5, 50, 0.8

CHIPS = [
    "Inception",
    "The Dark Knight",
    "Interstellar",
    "Spirited Away",
]

GENRE_SHORT = {
    "Science Fiction": "Sci-Fi",
    "Animation": "Animation",
    "Documentary": "Docu",
}


st.set_page_config(
    page_title="What2Watch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading movies...")
def load():

    df = pd.read_csv(
        ART / "movies_slim_v2.csv"
    )

    ov = pd.read_csv(
        DS / "tmdb_movies_25k.csv",
        usecols=["id", "overview"]
    ).drop_duplicates("id")

    # Left merge keeps row order = matrix order
    df = df.merge(
        ov,
        on="id",
        how="left"
    )

    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")

    X = joblib.load(
        ART / "tfidf_matrix_v2.pkl"
    )

    # Duplicate movie labels -> make them unique
    dup = df["label"].duplicated(keep=False)

    df["option"] = np.where(
        dup,
        df["label"] + " #" + df["id"].astype(str),
        df["label"]
    )

    order = (
        df.sort_values(
            "vote_count",
            ascending=False
        )["option"]
        .tolist()
    )

    return df, X, order


df, X, OPTIONS = load()


def find_option(title):

    m = (
        df[
            df["title"].str.lower()
            == title.lower()
        ]
        .sort_values(
            "vote_count",
            ascending=False
        )
    )

    return (
        m.iloc[0]["option"]
        if not m.empty
        else None
    )


def poster_of(title):

    m = (
        df[
            df["title"].str.lower()
            == title.lower()
        ]
        .sort_values(
            "vote_count",
            ascending=False
        )
    )

    if (
        m.empty
        or not isinstance(
            m.iloc[0]["poster_path"],
            str
        )
    ):
        return ""

    return (
        IMG
        + m.iloc[0]["poster_path"]
    )


def recommend(idx):

    sims = linear_kernel(
        X[idx],
        X
    ).ravel()

    sims[idx] = -1

    pool = np.argsort(
        sims
    )[::-1][:POOL]

    p = df.iloc[pool].copy()

    p["score"] = (
        ALPHA * sims[pool]
        + (1 - ALPHA)
        * p["quality_score"].values
    )

    return (
        p.sort_values(
            "score",
            ascending=False
        )
        .head(N_RECS)
    )


# ---------------------------------------------------------------- feedback db
def _db():

    DATA.mkdir(
        exist_ok=True
    )

    c = sqlite3.connect(
        DATA / "feedback.db"
    )

    c.execute(
        "CREATE TABLE IF NOT EXISTS feedback("
        "movie_id INTEGER PRIMARY KEY,"
        "likes INTEGER DEFAULT 0,"
        "dislikes INTEGER DEFAULT 0)"
    )

    c.execute(
        "CREATE TABLE IF NOT EXISTS votes("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "movie_id INTEGER,"
        "vote TEXT,"
        "ts TEXT DEFAULT CURRENT_TIMESTAMP)"
    )

    return c


def get_counts(mid):

    with closing(_db()) as c:

        r = c.execute(
            "SELECT likes, dislikes "
            "FROM feedback "
            "WHERE movie_id=?",
            (mid,)
        ).fetchone()

    return r or (0, 0)


def cast_vote(mid, kind):

    col = (
        "likes"
        if kind == "like"
        else "dislikes"
    )

    with closing(_db()) as c:

        c.execute(
            "INSERT OR IGNORE "
            "INTO feedback(movie_id) "
            "VALUES(?)",
            (mid,)
        )

        c.execute(
            f"UPDATE feedback "
            f"SET {col}={col}+1 "
            f"WHERE movie_id=?",
            (mid,)
        )

        c.execute(
            "INSERT INTO votes(movie_id, vote) "
            "VALUES(?,?)",
            (mid, kind)
        )

        c.commit()

    st.session_state.voted = kind


# ---------------------------------------------------------------- state
ss = st.session_state

ss.setdefault(
    "page",
    "home"
)

ss.setdefault(
    "voted",
    None
)

ss.setdefault(
    "movie",
    None
)


def go_recommend():

    opt = ss.get("sel")

    if not opt:

        st.toast(
            "Pick a movie from the list first.",
            icon="🎬"
        )

        return

    ss.movie = opt
    ss.voted = None
    ss.page = "results"


def pick_chip(title):

    ss["sel"] = find_option(title)


def go_home():

    ss.page = "home"
    ss.voted = None
    ss["sel"] = None


# ---------------------------------------------------------------- styling
@st.cache_data
def bg_css(name):

    for ext in (
        "jpg",
        "jpeg",
        "png",
        "webp"
    ):

        p = ASSETS / f"{name}.{ext}"

        if p.exists():

            mime = (
                "jpeg"
                if ext in ("jpg", "jpeg")
                else ext
            )

            b = base64.b64encode(
                p.read_bytes()
            ).decode()

            return (
                f"url(data:image/"
                f"{mime};base64,{b})"
            )

    return (
        "radial-gradient("
        "circle at 80% 30%, "
        "#3a1d10 0%, "
        "#0b0d14 60%)"
    )


CSS = """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap'
);


/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

html,
body,
[class*="st-"],
.stApp {
    font-family: 'Inter', sans-serif;
}


#MainMenu,
footer,
header[data-testid="stHeader"],
[data-testid="stToolbar"] {
    display: none !important;
}


.stApp {
    background:
        linear-gradient(
            90deg,
            rgba(6,8,14,.92) 0%,
            rgba(6,8,14,.55) 55%,
            rgba(6,8,14,.25) 100%
        ),
        BG center / cover fixed no-repeat;

    color: #f3f3f5;
}


.block-container {
    max-width: 1400px;
    padding: .8rem 2.5rem 1.5rem;
}


/* ---------------------------------------------------------
   LOGO
--------------------------------------------------------- */

.logo {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -.5px;
    margin-bottom: 0.6rem
}


.logo b {
    color: #ff2d3d;
    font-weight: 700;
}


/* ---------------------------------------------------------
   TEXT
--------------------------------------------------------- */

.eyebrow {
    letter-spacing: .27em;
    font-size: .68rem;
    color: #b9bcc6;
    text-transform: uppercase;
}


.grad {
    background:
        linear-gradient(
            90deg,
            #ff9a3c,
            #ff4d6d 60%,
            #ff7bb0
        );

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}


/* ---------------------------------------------------------
   HOMEPAGE
--------------------------------------------------------- */

.home-left {
    padding-top: 3.6rem;
}


.hero {
    font-size: 3.55rem;
    font-weight: 800;
    line-height: 1.02;
    letter-spacing: -1.8px;
    margin: .95rem 0 1.25rem;
}


.sub {
    color: #c9ccd6;
    font-size: 1rem;
    line-height: 1.5;
    max-width: 32rem;
    margin-bottom: 1.5rem;
}


/* ---------------------------------------------------------
   SELECTBOX
--------------------------------------------------------- */

div[data-baseweb="select"] > div {

    background:
        rgba(14,17,26,.75);

    border:
        1px solid
        rgba(255,255,255,.28);

    border-radius:
        999px;

    min-height:
        3.05rem;

    padding-left:
        .8rem;

    backdrop-filter:
        blur(8px);
}


div[data-baseweb="select"] * {
    color:
        #e8e9ee !important;
}


/*
   Keep the dropdown reasonably short so there is
   more space for it to open BELOW the selectbox.
*/
div[data-baseweb="menu"] {
    max-height:
        250px !important;

    overflow-y:
        auto !important;
}


div[data-baseweb="menu"] li {
    min-height:
        2.35rem !important;
}


ul[role="listbox"] {
    max-height:
        250px !important;

    overflow-y:
        auto !important;
}


/* Keep dropdown above other page elements */
div[data-baseweb="popover"] {
    z-index:
        999999 !important;
}


/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.st-key-go button {

    background:
        #ff4757;

    border:
        0;

    color:
        #fff;

    border-radius:
        999px;

    height:
        3.05rem;

    width:
        100%;

    font-weight:
        600;

    font-size:
        .95rem;
}


.st-key-go button:hover {

    background:
        #ff5f6d;

    box-shadow:
        0 0 24px
        rgba(255,71,87,.5);

    color:
        #fff;
}


.st-key-chips button,
.st-key-back button {

    background:
        rgba(20,24,34,.7);

    border:
        1px solid
        rgba(255,255,255,.16);

    color:
        #e6e7ec;

    border-radius:
        999px;

    padding:
        .1rem .85rem;

    min-height:
        2.15rem;

    font-size:
        .82rem;
}


.st-key-chips button:hover,
.st-key-back button:hover {

    border-color:
        #ff7a59;

    color:
        #fff;
}


.try {

    color:
        #9a9eab;

    padding-top:
        .33rem;

    font-size:
        .84rem;
}


/* ---------------------------------------------------------
   HOME POSTER STACK
--------------------------------------------------------- */

.stack {

    position:
        relative;

    height:
        415px;

    margin-top:
        48px;
}


.stack img {

    position:
        absolute;

    border-radius:
        16px;

    object-fit:
        cover;

    box-shadow:
        0 25px 60px
        rgba(0,0,0,.6);

    border:
        1px solid
        rgba(255,190,140,.35);
}


.p1 {

    width:
        180px;

    height:
        270px;

    left:
        4px;

    top:
        96px;

    transform:
        rotate(-3deg);

    filter:
        brightness(.8);
}


.p2 {

    width:
        245px;

    height:
        368px;

    left:
        158px;

    top:
        20px;

    z-index:
        3;

    transform:
        rotate(2deg);

    box-shadow:
        0 0 50px
        rgba(255,140,60,.3),
        0 20px 45px
        rgba(0,0,0,.6);
}


.p3 {

    width:
        180px;

    height:
        270px;

    left:
        382px;

    top:
        88px;

    z-index:
        2;

    transform:
        rotate(3deg);

    filter:
        brightness(.85);
}


/* ---------------------------------------------------------
   RESULTS PAGE
--------------------------------------------------------- */


/* Premium font for Recommendations */
.recommendation-label {

    font-family:
        'Playfair Display',
        serif;

    font-size:
        1.05rem;

    font-weight:
        600;

    letter-spacing:
        .34em;

    color:
        #d8d9df;

    text-transform:
        uppercase;

    text-align:
        center;

    margin-top:
        .8rem;
}


.rtitle {

    text-align:
        center;

    font-size:
        3.3rem;

    font-weight:
        700;

    letter-spacing:
        -1.5px;

    line-height:
        1.05;

    margin:
        .75rem 0 .65rem;
}


.rsub {

    text-align:
        center;

    letter-spacing:
        .25em;

    font-size:
        .78rem;

    color:
        #b9bcc6;

    margin-bottom:
        2.25rem;
}


/* ---------------------------------------------------------
   RESULT CARDS
--------------------------------------------------------- */

.card {

    background:
        rgba(10,12,20,.72);

    border:
        1px solid
        rgba(255,255,255,.12);

    border-radius:
        18px;

    overflow:
        hidden;

    height:
        600px;

    min-height:
        600px;

    max-height:
        600px;

    backdrop-filter:
        blur(6px);

    display:
        flex;

    flex-direction:
        column;
}


.ph {

    position:
        relative;

    height:
        330px;

    min-height:
        330px;

    max-height:
        330px;

    overflow:
        hidden;
}


.ph img {

    width:
        100%;

    height:
        100%;

    object-fit:
        cover;
}


.ph::after {

    content:
        "";

    position:
        absolute;

    inset:
        0;

    background:
        linear-gradient(
            transparent 60%,
            rgba(10,12,20,.98)
        );
}


.noposter {

    height:
        100%;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    text-align:
        center;

    padding:
        1rem;

    background:
        #151926;

    color:
        #9a9eab;
}


.rank {

    position:
        absolute;

    top:
        14px;

    left:
        14px;

    z-index:
        2;

    background:
        rgba(15,18,28,.85);

    border-radius:
        12px;

    border:
        1px solid
        rgba(255,255,255,.2);

    padding:
        .45rem .7rem;

    font-size:
        .9rem;
}


.cb {

    padding:
        .85rem 1.2rem 1.35rem;

    margin-top:
        -.6rem;

    position:
        relative;

    z-index:
        2;

    flex:
        1;

    display:
        flex;

    flex-direction:
        column;
}


.cb h3 {

    font-size:
        1.35rem;

    font-weight:
        700;

    margin:
        0 0 .5rem;

    line-height:
        1.18;

    min-height:
        3.15rem;

    display:
        -webkit-box;

    -webkit-line-clamp:
        2;

    -webkit-box-orient:
        vertical;

    overflow:
        hidden;
}


.meta {

    color:
        #b5b8c4;

    font-size:
        .9rem;

    margin-bottom:
        .85rem;

    min-height:
        1.2rem;
}


.cb p {

    color:
        #c0c3ce;

    font-size:
        .92rem;

    line-height:
        1.55;

    margin:
        0;

    display:
        -webkit-box;

    -webkit-line-clamp:
        6;

    -webkit-box-orient:
        vertical;

    overflow:
        hidden;
}


/* ---------------------------------------------------------
   THIN LINE ABOVE FEEDBACK
--------------------------------------------------------- */

.feedback-line {

    width:
        92%;

    height:
        1px;

    background:
        rgba(255,255,255,.13);

    margin:
        1.25rem auto 1.35rem;
}


/* ---------------------------------------------------------
   FEEDBACK
--------------------------------------------------------- */

.fbar h4 {

    font-size:
        1.6rem;

    font-weight:
        600;

    margin:
        0 0 .3rem;

    line-height:
        1.2;
}


.fbar span {

    color:
        #b5b8c4;
}


.st-key-like button,
.st-key-dislike button {

    height:
        4.2rem;

    width:
        100%;

    border-radius:
        999px;

    font-weight:
        500;
}


.st-key-like button {

    background:
        rgba(20,60,45,.45);

    border:
        1px solid
        #2f8f6b;

    color:
        #d6fff0;
}


.st-key-dislike button {

    background:
        rgba(70,20,28,.45);

    border:
        1px solid
        #b0404f;

    color:
        #ffd9de;
}


.st-key-like button:hover {

    box-shadow:
        0 0 22px
        rgba(47,143,107,.5);
}


.st-key-dislike button:hover {

    box-shadow:
        0 0 22px
        rgba(176,64,79,.5);
}


.thanks {

    text-align:
        center;

    color:
        #d6fff0;

    padding:
        1rem 0 .2rem;

    font-size:
        1.05rem;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 900px) {

    .hero {
        font-size:
            2.7rem;
    }

    .home-left {
        padding-top:
            2rem;
    }

    .stack {
        display:
            none;
    }

    .block-container {
        padding:
            .8rem 1rem 1rem;
    }

    .card {

        height:
            560px;

        min-height:
            560px;

        max-height:
            560px;
    }

    .ph {

        height:
            300px;

        min-height:
            300px;

        max-height:
            300px;
    }

    .rtitle {
        font-size:
            2.4rem;
    }

    .logo {
        font-size:
            1.85rem;
    }
}

</style>
"""


def inject(bg):

    st.markdown(
        CSS.replace("BG", bg),
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------- pages
def home():

    inject(
        bg_css("bg_home")
    )

    st.markdown(
        '<div class="logo">'
        'What<b>2</b>Watch'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.1, 1],
        gap="medium"
    )

    with left:

        st.markdown(
            '<div class="home-left">'

            '<div class="eyebrow">'
            'Smart movie recommendation system'
            '</div>'

            '<div class="hero">'
            'Find Your<br>'
            '<span class="grad">'
            'Next Favorite'
            '</span> Movie'
            '</div>'

            '<div class="sub">'
            'Get personalised movie recommendations '
            'based on what you like. '
            'Enter a movie you enjoy and discover '
            'similar ones.'
            '</div>'

            '</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(
            [4, 1.25],
            gap="medium"
        )

        c1.selectbox(
            "Movie",
            OPTIONS,
            index=None,
            placeholder="Type a movie title...",
            key="sel",
            label_visibility="collapsed",
        )

        with c2:

            with st.container(key="go"):

                st.button(
                    "Recommend",
                    on_click=go_recommend,
                    use_container_width=True
                )

        st.markdown(
            '<div style="height:.35rem;"></div>',
            unsafe_allow_html=True
        )

        with st.container(key="chips"):

            cols = st.columns(
                [.6, 1.2, 1.7, 1.5, 1.6, 2],
                gap="small"
            )

            cols[0].markdown(
                '<div class="try">Try</div>',
                unsafe_allow_html=True
            )

            for col, t in zip(
                cols[1:5],
                CHIPS
            ):

                col.button(
                    t,
                    key=f"chip_{t}",
                    on_click=pick_chip,
                    args=(t,),
                    use_container_width=True
                )

    with right:

        u = [
            poster_of("Interstellar"),
            poster_of("Inception"),
            poster_of("The Dark Knight")
        ]

        st.markdown(
            '<div class="stack">'
            +
            "".join(
                f'<img class="p{i + 1}" src="{src}">'
                for i, src in enumerate(u)
                if src
            )
            +
            "</div>",
            unsafe_allow_html=True
        )


def card(rank, r):

    gs = [
        GENRE_SHORT.get(
            g.strip(),
            g.strip()
        )
        for g in r.genres.split(",")
        if g.strip()
    ][:2]

    meta = " · ".join(
        [str(int(r.year))] + gs
    )

    poster = (
        f'<img src="{IMG}{r.poster_path}">'
        if isinstance(
            r.poster_path,
            str
        )
        else
        f'<div class="noposter">'
        f'{html.escape(r.title)}'
        f'</div>'
    )

    ov = (
        html.escape(r.overview)
        or "No description available."
    )

    return (

        f'<div class="card">'

        f'<div class="ph">'

        f'{poster}'

        f'<span class="rank">'
        f'{rank:02d}'
        f'</span>'

        f'</div>'

        f'<div class="cb">'

        f'<h3>'
        f'{html.escape(r.title)}'
        f'</h3>'

        f'<div class="meta">'
        f'{meta}'
        f'</div>'

        f'<p>'
        f'{ov}'
        f'</p>'

        f'</div>'

        f'</div>'
    )


def results():

    inject(
        bg_css("bg_results")
    )

    idx = int(
        df.index[
            df["option"] == ss.movie
        ][0]
    )

    row = df.iloc[idx]

    mid = int(row["id"])

    st.markdown(
        '<div class="logo" '
        'style="margin-bottom:1.2rem">'
        'What<b>2</b>Watch'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(key="back"):

        st.button(
            "←  Back to Home",
            on_click=go_home
        )

    st.markdown(

        '<div class="recommendation-label">'
        'Recommendations'
        '</div>'

        f'<div class="rtitle">'
        'Because you liked '
        f'<span class="grad">'
        f'{html.escape(row.title)}'
        f'</span>'
        '</div>'

        '<div class="rsub">'
        'HERE ARE SOME SIMILAR MOVIES '
        'YOU MIGHT ENJOY.'
        '</div>',

        unsafe_allow_html=True
    )

    recs = recommend(idx)

    cols = st.columns(
        N_RECS,
        gap="small"
    )

    for col, (rank, r) in zip(
        cols,
        enumerate(
            recs.itertuples(),
            1
        )
    ):

        with col:

            st.markdown(
                card(rank, r),
                unsafe_allow_html=True
            )

    # ---------------------------------------------------------
    # Feedback separator
    # ---------------------------------------------------------

    st.markdown(
        '<div class="feedback-line"></div>',
        unsafe_allow_html=True
    )

    likes, dislikes = get_counts(mid)

    # ---------------------------------------------------------
    # Feedback section
    # No manually opened/closed HTML div here.
    # This removes the empty rounded rectangle.
    # ---------------------------------------------------------

    a, b, c = st.columns(
        [1.2, 1, 1],
        gap="medium",
        vertical_alignment="center"
    )

    a.markdown(

        f"<h4>"
        f"How do you feel about<br>"
        f"these recommendations?"
        f"</h4>"

        f"<span>"
        f"👍 {likes} · 👎 {dislikes} "
        f"people rated the picks for this movie."
        f"</span>",

        unsafe_allow_html=True
    )

    done = ss.voted is not None

    with b:

        with st.container(key="like"):

            st.button(
                "👍  These are good recommendations",
                on_click=cast_vote,
                args=(mid, "like"),
                disabled=done,
                use_container_width=True
            )

    with c:

        with st.container(key="dislike"):

            st.button(
                "👎  Not my taste",
                on_click=cast_vote,
                args=(mid, "dislike"),
                disabled=done,
                use_container_width=True
            )

    if done:

        st.markdown(
            "<div class='thanks'>"
            "Thanks for the feedback! "
            "Pick another movie from Back to Home."
            "</div>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------- run
if ss.page == "home":
    home()
else:
    results()