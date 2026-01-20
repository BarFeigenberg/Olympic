import streamlit as st


def show_project_overview():
    # יצירת שתי עמודות: השמאלית לטקסט (תופסת 2/3 מהרוחב) והימנית ריקה (1/3)
    # זה שומר על יישור לשמאל ומונע מהשורות להיות ארוכות מדי
    left_column, right_spacer = st.columns([2, 1])

    with left_column:
        st.markdown("# Olympic Evolution (1896–2024)")
        st.markdown("### A Multi-Dimensional Analysis of Sporting Excellence")
        st.write("")

        st.markdown("""
        The Olympic Insights Dashboard transforms over a century of raw results into an interactive exploration of global sporting history. By integrating data from **Kaggle**, the **IOC**, and **Gapminder**, this platform analyzes the structural, socio-economic, and physical factors that define Olympic achievement from 1896 to 2024.

        ---

        ### Global Overview
        This section re-evaluates world achievements by looking beyond simple medal counts. It examines whether a massive population naturally leads to dominance or if normalized metrics reveal a hidden hierarchy of sporting efficiency. 
        
        *Navigate to **Global Overview** to explore the global map and individual country growth trends.*

        ### Host Advantage
        Investigating the statistical reality behind the "Home Field Advantage" allows us to see the true impact of hosting the Games. 

        *Navigate to **Host Advantage** to visualize the flow of performance and quantify the actual boost gained from hosting.*

        ### Athletics Deep Dive
        Excellence in athletics often raises questions about the physical attributes of champions. Here, we examine the correlation between height, weight, and medal success. 

        This page allows you to identify the physiological profiles of winners and observe how Olympic records have evolved across genders throughout history. 

        *Navigate to **Athletics Deep Dive** to explore the science of records and the physical build of the world's elite athletes.*

        ### Wellness & Winning
        The final layer of our analysis explores the intersection of national health, wealth, and sporting success. 

        By correlating factors such as life expectancy and resource allocation (delegation size) with medal output, we can observe how national development impacts Olympic results over time. 

        *Navigate to **Wellness & Winning** to witness a dynamic animation of nations navigating their path toward the podium.*
        """)