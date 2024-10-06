import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from scipy.io import arff

# --- File Processing Functions ---

def convert_to_csv(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer

def read_uploaded_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
    
    try:
        if file_extension == '.csv':
            return pd.read_csv(uploaded_file)
        elif file_extension == '.xlsx':
            df = pd.read_excel(uploaded_file)
            return pd.read_csv(convert_to_csv(df))
        elif file_extension == '.arff':
            decoded_file = uploaded_file.read().decode('utf-8')
            data, meta = arff.loadarff(StringIO(decoded_file))
            df = pd.DataFrame(data)
            
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            
            return pd.read_csv(convert_to_csv(df))
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# --- Plotting Functions ---

def create_plot(df, x_axis, y_axis, plot_type, plot_title):
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'Line Plot':
        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
    elif plot_type == 'Bar Chart':
        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
    elif plot_type == 'Scatter Plot':
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
    elif plot_type == 'Distribution Plot':
        sns.histplot(df[x_axis], kde=True, ax=ax)
        y_axis = 'Density'
    elif plot_type == 'Count Plot':
        sns.countplot(x=df[x_axis], ax=ax)
        y_axis = 'Count'

    ax.tick_params(axis='both', labelsize=10)
    plt.title(plot_title, fontsize=12)
    plt.xlabel(x_axis, fontsize=10)
    plt.ylabel(y_axis, fontsize=10)

    return fig

# --- Main App ---

def main():
    st.set_page_config(page_title='Data Visualizer', layout='centered', page_icon='📊')
    st.title('📊 Data Visualizer')

    uploaded_file = st.file_uploader("Upload a file", type=['csv', 'arff', 'xlsx'])

    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        
        if df is not None:
            display_data_and_controls(df)

def display_data_and_controls(df):
    col1, col2 = st.columns(2)

    with col1:
        st.write("Data Preview")
        st.dataframe(df, height=400)

    with col2:
        columns = df.columns.tolist()
        x_axis = st.selectbox('Select the X-axis', options=columns + ["None"])
        y_axis = st.selectbox('Select the Y-axis', options=columns + ["None"])

        plot_list = ['Line Plot', 'Bar Chart', 'Scatter Plot', 'Distribution Plot', 'Count Plot']
        plot_type = st.selectbox('Select the type of plot', options=plot_list)

        plot_title = st.text_input("Enter the plot title", value=f'{plot_type} of {y_axis} vs {x_axis}')

    if st.button('Generate Plot'):
        fig = create_plot(df, x_axis, y_axis, plot_type, plot_title)
        st.pyplot(fig)

        if st.button('Download Plot'):
            fig.savefig('plot.png')
            st.success("Plot saved as plot.png")

    display_pivot_table_controls(df, columns)

def display_pivot_table_controls(df, columns):
    st.title('📊 Pivot Table')
    index_col = st.selectbox("Select Index Column", options=columns)
    value_col = st.selectbox("Select Value Column", options=columns)
    
    agg_func = st.selectbox("Select Aggregation Function", options=["mean", "sum", "count"])

    if agg_func in ["mean", "sum"] and not pd.api.types.is_numeric_dtype(df[value_col]):
        st.warning("Cannot use mean or sum on non-numeric data.")
    elif st.button("Generate Pivot Table"):
        pivot_table = pd.pivot_table(df, index=index_col, values=value_col, aggfunc=agg_func)
        st.dataframe(pivot_table)

if __name__ == "__main__":
    main()