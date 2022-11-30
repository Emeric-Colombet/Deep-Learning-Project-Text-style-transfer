import streamlit as st
from annotated_text import annotated_text

st.set_page_config(
    page_title="¡hola!",
    page_icon="👋",
    layout="wide"
)

st.image('assets/hola.jpeg')

st.write("# Welcome to the Spanish Style Transfer App! 👋")

st.markdown(
"""
##### Our goal is to turn the neutral style of Latin-American Spanish text into the style of European Spanish.
👈 Go to <a href="http://localhost:8111/Running_Style_Transfer_%F0%9F%93%96" target="_self">Running Style Transfer</a> on the sidebar to start!

#
#
### What is Latin-American Spanish?
Latin America is a group of countries that stretch from the northern border of Mexico to the southern tip of South America. 
Because the territory is so large, there is no “uniform” Spanish as each country’s dialect is unique and varies greatly. 

In order to cater to the majority of Latin American Spanish speakers, translators developed what is referred to as “LATAM Spanish”. 
This neutral or generic Spanish avoids country colloquialisms but still sounds familiar with the general audience. 

#
#
### Key differences in between Spanish styles
###### Spanish is the official language in 21 countries, but it isn’t the same worldwide. The accent, pronunciation, grammar and vocabulary varies from one country to another
""",
    unsafe_allow_html=True
)

col_left, col_right = st.columns([10, 10])

with col_left:
    st.markdown(
"""
##### ✅ OUR MAIN SCOPE WITH THE APP

**1. Vocabulary**

Tons of variation in vocabulary. 
For example, depending on where they’re from, Spanish speakers may have different words for beer 🍺:
caña 🇪🇸, cerveza 🇵🇷, chela 🇲🇽, pola 🇨🇴, birra 🇦🇷
¡Salud!

**2. Vosotros vs ustedes (you plural)**
- English: "Are you all friends? Where did you meet?"
- Latin America: "¿**Ustedes son** amigos? ¿Dónde **se conocieron**?"
- Spain: "¿**Vosotros sois** amigos? ¿Dónde **os conocisteis**?"
- Spain formal: "¿Quieren [**ustedes**] algo de comer?"

**3. Past tense**
- Latin American: single past: "Hoy **fui** al trabajo" (I **went** to work today)
- European Spanish: present perfect: "Hoy **he ido** al trabajo" (I **have gone** to work today)
"""
)

with col_right:
    st.markdown(
"""
##### ❌ OUT OF SCOPE FOR THE APP

**4. Accent and pronunciation differences**
- Z and C: **[s]** apato vs **[th]** apato 👞
- S at the end of words: gracia **[h]** vs gracia **[s]** 🙏
- LL and Y: **[j]** uvia vs **[sh]** uvia ☔
- J: **[j]** amón vs **[JJJJ]** amón 🐖

**5. Tú vs usted vs vos (you singular)**
- Tú: informal
- Usted: formal
- Vos: Argentina
"""
)

st.markdown(
"""
#
#
### Some examples
"""
)

col_latinamerica, col_spain = st.columns([10, 10])

with col_latinamerica:
    st.markdown('##### 🌎 LATINAMERICA')

    st.markdown(
        """
        <div style="height: 0; padding-bottom: calc(56.25%); position:relative; width: 100%;">
            <iframe allow="autoplay; gyroscope;" allowfullscreen height="100%" referrerpolicy="strict-origin" 
            src="https://www.kapwing.com/e/6387c24312c6de0046e2b898" style="border:0; height:100%; left:0; 
            overflow:hidden; position:absolute; top:0; width:100%" title="Embedded content made on Kapwing" width="100%">
            </iframe>
        </div>
        """,
    unsafe_allow_html=True
    )

    annotated_text('¡Ya me harté!')
    annotated_text('¡Esta es mi fiesta y a ', ('nadie', 'you plural'), ('le importa!', 'colloquial'))
    annotated_text('Escucha, tú te portas como una histérica,')
    annotated_text('tú siempre estas ahí con el ', ('celular', 'vocabulary'), 'como si lo acabaran de inventar.')
    annotated_text('Oigan,')
    annotated_text('¿no se les ', ('olvida', 'tense'), 'algo?')
    annotated_text('Mi tesis, ¡10 años de estudio!')
    annotated_text('Y por favor no me hagan hablar de Manon.')
    annotated_text('Francamente, ¿alguien sabe dónde está?')
    annotated_text('Pues bueno, no lo sabemos.')
    annotated_text('Siento que a ', ('nadie', 'you plural'), 'le importa que me haya convertido en doctor.')
    annotated_text('¿Por qué? ¿No es importante verdad? ¡Doctor, un ingeniero de vida!')
    annotated_text('Era mi momento, ¡mi momento!')

with col_spain:
    st.markdown('##### 🇪🇸 SPAIN')

    st.markdown(
        """
        <div style="height: 0; padding-bottom: calc(56.25%); position:relative; width: 100%;">
            <iframe allow="autoplay; gyroscope;" allowfullscreen height="100%" referrerpolicy="strict-origin" 
            src="https://www.kapwing.com/e/6387c0498215b700838e0588" style="border:0; height:100%; left:0; 
            overflow:hidden; position:absolute; top:0; width:100%" title="Embedded content made on Kapwing" width="100%">
            </iframe>
        </div>
        """,
    unsafe_allow_html=True
    )

    annotated_text('¡Ya basta!')
    annotated_text('¡Es mi tesis y a todos ', ('os', 'you plural'), ('la suda!', 'colloquial'))
    annotated_text('Entre tú que te comportas como una histérica,')
    annotated_text('tú todo el día con el ', ('móvil', 'vocabulary'), 'como si lo acabaran de inventar.')
    annotated_text('Y tú Yacine, ¡hola Yacine!')
    annotated_text('¿No te parece que te ', ('has perdido', 'tense'), 'algo?')
    annotated_text('Mi tesis, ¡10 años de estudios!')
    annotated_text('Por no hablar de Manon.')
    annotated_text('Por cierto, ¿dónde está?')
    annotated_text('Lo ', ('veis', 'you plural'), 'ni se sabe,')
    annotated_text('parece que ', ('os', 'you plural'), 'importa ', ('un cuerno', 'colloquial'), 'a todos que sea doctor.')
    annotated_text('Muy bien, y no es poca cosa doctor, ¡ingeniero de vida!')
    annotated_text('Y éste era mi momento, ¡mi momento!')
