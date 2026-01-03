import streamlit as st
import plotly.express as px


def show_wellness_winning(gap_df):
    st.title("📈 Wellness & Winning Over Time")
    if gap_df is not None and not gap_df.empty:
        roi_mode = st.radio("Select View:", ["Efficiency (Medals per Million)", "Total Impact (Total Medals)"],
                            horizontal=True)

        if roi_mode == "Efficiency (Medals per Million)":
            fig = px.scatter(gap_df, x="Life_Expectancy", y="Medals_Per_Million", animation_frame="Year",
                             animation_group="Country_Name", size="Delegation_Size", color="continent",
                             hover_name="Country_Name", size_max=50, range_x=[35, 90], range_y=[-0.5, 6],
                             title="1. Health vs Talent (Medals Per Million)")
        else:
            gap_df['Medals_Log_Proxy'] = gap_df['Medals'].replace(0, 0.5)
            fig = px.scatter(gap_df, x="Life_Expectancy", y="Medals_Log_Proxy", animation_frame="Year",
                             animation_group="Country_Name", size="Delegation_Size", color="continent",
                             hover_name="Country_Name", hover_data=["Medals"], size_max=50, range_x=[35, 90],
                             range_y=[0.4, 450], log_y=True, title="2. Health vs Total Medals (Log Scale)")
            fig.update_layout(yaxis=dict(title="Total Medals (Log Scale)", tickvals=[0.5, 1, 5, 10, 50, 100, 400],
                                         ticktext=["0", "1", "5", "10", "50", "100", "400"]))

        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
        st.plotly_chart(fig, use_container_width=True)
