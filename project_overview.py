import streamlit as st


def show_project_overview():
    # HERO SECTION
    st.markdown(
        """
        <div style="max-width: 900px; margin-bottom: 2rem;">
            <h1>Olympic Evolution (1896–2024)</h1>
            <p style="font-size:18px; color:#6c757d;">
                A Multi-Dimensional Analysis of Global Sporting Excellence
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # INTRO
    st.markdown(
        """
        <div style="max-width: 900px; font-size:16px; line-height:1.6;">
        The Olympic Insights Dashboard transforms over a century of Olympic results into an interactive exploration of global sporting history.
        By integrating data from Kaggle, the IOC, and Gapminder, the project examines how demographic, economic, and physical factors shape Olympic success from 1896 to 2024.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # SECTION TEMPLATE STYLE
    section_style = """
    <div style="
        max-width: 900px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 10px;
        background-color: rgba(230, 236, 244, 0.6);
        border: 1px solid rgba(0, 0, 0, 0.08);
    ">
        <h2>{title}</h2>
        <p>{p1}</p>
        <p>{p2}</p>
        <p style="font-size:14px; color:#6c757d;">{nav}</p>
    </div>
    """

    # SECTION 1
    st.markdown(
        section_style.format(
            title="Global Overview",
            p1="A re-examination of global Olympic dominance beyond raw medal counts.",
            p2="This view explores whether population size truly predicts success, or if normalized performance metrics reveal a hidden hierarchy of sporting efficiency.",
            nav="Navigate to Global Overview to explore global maps and country-level growth trends."
        ),
        unsafe_allow_html=True
    )

    # SECTION 2
    st.markdown(
        section_style.format(
            title="Host Advantage",
            p1="An empirical investigation into the competitive impact of hosting the Olympic Games.",
            p2="This section quantifies performance shifts across Olympic cycles and measures the actual competitive boost gained by host nations.",
            nav="Navigate to Host Advantage to explore performance flows across hosting periods."
        ),
        unsafe_allow_html=True
    )

    # SECTION 3
    st.markdown(
        section_style.format(
            title="Athletics Deep Dive",
            p1="A focused analysis of athletic performance and the physical attributes of elite competitors.",
            p2="The analysis examines relationships between height, weight, gender, and medal success, while tracking the evolution of Olympic records over time.",
            nav="Navigate to Athletics Deep Dive to explore record progression and athlete physiology."
        ),
        unsafe_allow_html=True
    )

    # SECTION 4
    st.markdown(
        section_style.format(
            title="Wellness & Winning",
            p1="An exploration of the relationship between national development and Olympic performance.",
            p2="By correlating life expectancy, economic indicators, and delegation size with medal output, this view illustrates how health and wealth influence success on the Olympic stage.",
            nav="Navigate to Wellness & Winning to observe long-term development trends and medal outcomes."
        ),
        unsafe_allow_html=True
    )
