import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import geopandas as gpd
from streamlit_dynamic_filters import DynamicFilters
import folium
from folium.plugins import MarkerCluster
import calendar
from PIL import Image

# Streamlit App Layout
st.set_page_config(layout="wide")


# Load the dataset
# Replace this with your actual dataset
@st.cache_data
def load_data():
    # Load your Airbnb dataset here
    df = pd.read_csv('myenv/listing_df1.csv')
    df=df.drop(columns='Unnamed: 0')
    df.drop(columns=['Street','Last Review','City','State','Id','Host Id'],inplace=True)
    return df

df = load_data()

# Function to filter data based on user selections
def filter_data(df, city=None, state=None, area=None,price_range=None,availability_range=None):
    if area:
        df=df[df['Government Area'].isin(area)]
    if city:
        df = df[df['Suburban'].isin(city)]
    if state:
        df = df[df['Market'] == state]
    if price_range:
        df = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]
    if availability_range:
        df = df[(df['Availability_365'] >= availability_range[0]) & (df['Availability_365'] <= availability_range[1])]
    return df



st.markdown("""
    <style>
    /* Adjust sidebar */
    .css-1d391kg {
        font-size: 22px !important;  /* Larger font size */
        font-family: 'Arial', sans-serif;  /* Font family for better look */
        font-weight: bold;
    }
    
    /* Adjust font sizes of radio buttons or option menu */
    .css-1v3fvcr {
        font-size: 22px !important;
        font-weight: bold;
    }
    
    /* Adjust the main menu container style */
    .css-1v2xy1a {
        font-size: 28px;
        font-weight: bold;
        color: #f0f0f0;
    }

    /* Adjust the background of the sidebar */
    .css-1d391kg {
        background-color: #333333 !important;
    }

    /* Custom icons */
    .stRadio label {
        font-size: 24px !important;
        display: flex;
        align-items: center;
    }

    /* Customize hover effect */
    .stRadio label:hover {
        background-color: #444444;
        border-radius: 10px;
        padding: 5px;
    }

    </style>
""", unsafe_allow_html=True)

# Sidebar with icons and larger font
with st.sidebar:
    st.title("Main Menu")  # Larger Title for Sidebar
    selected = option_menu(
        "Go to", ["🏠 Home", "📊 EDA", "📞 Contact"],
        icons=['house', 'bar-chart', 'phone'],  # Icons for the options
        menu_icon="cast", default_index=0, orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#333"},
            "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#444"},
            "nav-link-selected": {"background-color": "#006eff"},
        }
    )

    # Navigation Logic
if selected == "🏠 Home":
    st.title("Home")
    st.write("Welcome to the Airbnb App Home Page")
     # Add a brief description or summary
    st.write("This dashboard provides insights into Airbnb listings and their key metrics such as price analysis, availability, and geographic distribution. Use the navigation on the left to explore different sections.")
    
    # Dynamic Section with Image
    image = Image.open('airbnbi.png')  # Replace with your image path
    st.image(image, caption="Explore Airbnb data visually!")

    # Add dynamic updates or alerts
    st.markdown("### Latest Updates")
    st.markdown("### Key Metrics")
    st.metric(label="Total Listings", value="5,555")
    st.metric(label="Average Price", value="270")
    st.metric(label="Highest Rating", value="100")
    st.info("Stay tuned for the latest analysis and insights.")

elif selected == "📊 EDA":
    st.title("Exploratary analysis")
    st.write("Welcome to the Analysis & Visualization")



    # Inject custom CSS to style the tabs
    st.markdown("""
        <style>
        /* Style for the navigation tabs */
        .css-18e3th9 {
            padding-top: 10px;
        }
        .css-1v3fvcr {
            background-color: #121212;
            color: white;
        }
        /* Tab text styles */
        .stTabs ul li a {
            padding: 100px;
            margin-right: 100px;
            color: #fff;
            font-weight: bold;
            background: #444444;
            border-radius: 5px;
            transition: background 0.2s ease-in-out;
        }
        .stTabs ul li a:hover {
            background: #888888;
        }
        /* Selected tab */
        .stTabs ul li a[aria-selected="true"] {
            background: #FF4B4B;
            color: #fff;
        }
        </style>"""
    , unsafe_allow_html=True)

    # Add tabs
    tabs = st.tabs(["Price Analysis", "Availability Analysis", "Location Insights", "Interactive Map"])

    with tabs[0]:
        st.write("Price Trends Across Locations")

        #Group by

        # Group by Neighborhood to find average prices
        
        price_by_City= df.groupby('Suburban')['Price'].mean().reset_index()

        price_by_State=df.groupby('Market')['Price'].mean().reset_index()

        # Group by Property Type
        price_by_property_type = df.groupby('Property Type')['Price'].mean().reset_index()

        price_by_Room_type=df.groupby('Room Type')['Price'].mean().reset_index()

                
        col1,col2=st.columns(2)

        
        with col1:

            fig_City=px.bar(price_by_City,x="Suburban",y='Price',title="Price By City ",color_discrete_sequence=px.colors.sequential.Redor_r,height=600,width=600)
            st.plotly_chart(fig_City)

        with col2:

            fig_Sta=px.bar(price_by_State,x="Market",y='Price',title="Price By State",color_discrete_sequence=px.colors.sequential.Bluered_r,height=600,width=600)
            st.plotly_chart(fig_Sta)


        # Tab for Price Analysis
        def price_analysis(df):
            st.header("Price Analysis Across Locations")

            # Filter options
            Area=df['Government Area'].unique()
            cities = df['Suburban'].unique()
            states = df['Market'].unique()
            
        

            # Create a selectbox for states
            selected_state = st.selectbox("Filter by State", ["All States"] + list(states))

            # Filter cities and areas based on the selected state
            if selected_state == "All States":
                filtered_cities = cities  # If no state is selected, show all cities
                filtered_areas = Area     # Show all areas
            else:
                filtered_cities = df[df['State'] == selected_state]['Suburban'].unique()  # Filter cities by state
                filtered_areas = df[df['State'] == selected_state]['Government Area'].unique()  # Filter areas by state

            # Create multi-select boxes for cities and areas based on the selected state
            selected_cities = st.multiselect("Filter by City", filtered_cities)
            selected_area = st.multiselect("Filter by Area", filtered_areas)

            # Slider to filter price range
            price_range = st.slider("Select Price Range", int(df['Price'].min()), int(df['Price'].max()), (50, 500))

            # Filter data based on user input
            filtered_data = filter_data(df, area=selected_area,city=selected_cities, state=selected_state if selected_state != "All States" else None, price_range=price_range)

            # 1. Price Distribution (Histogram)
            st.subheader("Price Distribution")
            fig, ax = plt.subplots()
            sns.histplot(filtered_data['Price'], bins=30, ax=ax, kde=True)
            ax.set_title("Price Distribution")
            st.pyplot(fig)

            # 2. Price by City (Bar Chart)
            st.subheader("Average Price by City")
            avg_price_city = filtered_data.groupby('Suburban')['Price'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(avg_price_city, x=avg_price_city.index, y=avg_price_city.values, labels={'x': 'City', 'y': 'Average Price'}, title="Price by City")
            st.plotly_chart(fig)

            # 3. Price by State (Bar Chart)
            st.subheader("Average Price by State")
            avg_price_state = filtered_data.groupby('Market')['Price'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(avg_price_state, x=avg_price_state.index, y=avg_price_state.values, labels={'x': 'State', 'y': 'Average Price'}, title="Price by State")
            st.plotly_chart(fig)

            # 4. Price by Property Type
            st.subheader("Average Price by Property Type")
            avg_price_property = filtered_data.groupby('Property Type')['Price'].mean().sort_values(ascending=False)
            fig = px.bar(avg_price_property, x=avg_price_property.index, y=avg_price_property.values, labels={'x': 'Property Type', 'y': 'Average Price'}, title="Price by Property Type")
            st.plotly_chart(fig)

            # 5. Price by Room Type
            st.subheader("Average Price by Room Type")
            avg_price_room = filtered_data.groupby('Room Type')['Price'].mean().sort_values(ascending=False)
            fig = px.bar(avg_price_room, x=avg_price_room.index, y=avg_price_room.values, labels={'x': 'Room Type', 'y': 'Average Price'}, title="Price by Room Type")
            st.plotly_chart(fig)

            # 6. Correlation Analysis (Price vs Reviews, Bedrooms, etc.)
            st.subheader("Correlation Analysis")
            corr_df = filtered_data[['Price', 'No of Reviews', 'Bedrooms', 'Rating']].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation between Price and Other Variables")
            st.pyplot(fig)

            # 7. Outliers in Price (Box Plot)
            st.subheader("Detecting Price Outliers")
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered_data, x='Price', ax=ax)
            ax.set_title("Price Outliers")
            st.pyplot(fig)

        # Place the price analysis function inside the Price Analysis tab
        price_analysis(df)




    with tabs[1]:
        st.write("Availability Trends")
        # Tab for Availability Analysis
        def availability_analysis(df):
            st.header("Availability Analysis Across Locations")

            # Filter options
            Urbans = df['Suburban'].unique()
            Markets = df['Market'].unique()

            # Create multi-select box for cities
            select_cities = st.multiselect("Filter by City", Urbans,key='City_select1')

            # Create a selectbox for states
            select_state = st.selectbox("Filter by State", ["All States"] + list(Markets),key='state_select1')

            # Slider to filter availability range (0-365 days)
            availability_range = st.slider("Select Availability Range (Days)", 0, 365, (0, 365))

            # Filter data based on user input
            filtered_data = filter_data(df, city=select_cities, state=select_state if select_state != "All States" else None, availability_range=availability_range)

            # 1. Availability Distribution (Histogram)
            st.subheader("Availability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(filtered_data['Availability_365'], bins=30, ax=ax, kde=True)
            ax.set_title("Availability Distribution (365 days)")
            st.pyplot(fig)

            # 2. Availability by City (Bar Chart)
            st.subheader("Average Availability by City")
            avg_avail_city = filtered_data.groupby('Suburban')['Availability_365'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(avg_avail_city, x=avg_avail_city.index, y=avg_avail_city.values, labels={'x': 'Suburban', 'y': 'Average Availability (Days)'}, title="Availability by City")
            st.plotly_chart(fig)

            # 3. Availability by State (Bar Chart)
            st.subheader("Average Availability by State")
            avg_avail_state = filtered_data.groupby('Market')['Availability_365'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(avg_avail_state, x=avg_avail_state.index, y=avg_avail_state.values, labels={'x': 'Market', 'y': 'Average Availability (Days)'}, title="Availability by State")
            st.plotly_chart(fig)

            # 4. Availability by Property Type
            st.subheader("Average Availability by Property Type")
            avg_avail_property = filtered_data.groupby('Property Type')['Availability_365'].mean().sort_values(ascending=False)
            fig = px.bar(avg_avail_property, x=avg_avail_property.index, y=avg_avail_property.values, labels={'x': 'Property Type', 'y': 'Average Availability (Days)'}, title="Availability by Property Type")
            st.plotly_chart(fig)

            # 5. Availability by Month (Seasonality Analysis)
            st.subheader("Seasonal Availability Analysis")

            # Filter out any invalid month values (like 0)
            filtered_data = df[df['Month'].notna() & (df['Month'] > 0) & (df['Month'] <= 12)]
            
            monthly_availability = filtered_data.groupby('Month')['Availability_365'].mean()

            fig, ax = plt.subplots()
            ax.plot(monthly_availability.index, monthly_availability.values, marker='o')
            ax.set_title("Average Availability by Month")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([calendar.month_name[i] for i in range(1, 13)])
            ax.set_ylabel("Average Availability (Days)")
            st.pyplot(fig)

            # 6. Booking Patterns (Heatmap)
            st.subheader("Availability Heatmap by Month")
            heatmap_data = filtered_data.pivot_table(values='Availability_365', index='Market', columns='Month', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_data, cmap="coolwarm", ax=ax)
            ax.set_title("Average Availability (Days) by Market and Month")
            st.pyplot(fig)

        # Place the availability analysis function inside the Availability Analysis tab
        availability_analysis(df)

    with tabs[2]:
        st.header("Location-Based Insights")
 
        # Function for Location Insights Tab
        def location_insights(df):
            st.title("Location-Based Insights")
            
            # Option 1: Filter by City
            cities = df['Suburban'].unique()
            select_city = st.multiselect("Select City", cities, key="city_select2")
            
            # Filter DataFrame based on the selected city
            if select_city:
                df = df[df['City'].isin(select_city)]
            
            # Option 2: Filter by Neighborhood (if applicable in the dataset)
            if 'Neighborhood' in df.columns:
                neighborhoods = df['Neighborhood'].unique()
                select_neighborhood = st.multiselect("Select Neighborhood", neighborhoods, key="neighborhood_select")
                
                # Filter DataFrame based on selected neighborhood
                if select_neighborhood:
                    df = df[df['Neighborhood'].isin(select_neighborhood)]
            
            # Option 3: Price Distribution by Location
            st.subheader("Price Distribution by City")
            fig_price_city = px.box(df, x='Suburban', y='Price', color='Suburban',
                                    title="Price Distribution by City",
                                    labels={'Price': 'Price (in currency)', 'Suburban': 'City'})
            st.plotly_chart(fig_price_city, use_container_width=True)
            
            # Option 4: Average Rating by City
            st.subheader("Average Rating by City")
            avg_rating_city = df.groupby('Suburban')['Rating'].mean().reset_index()
            fig_rating_city = px.bar(avg_rating_city, x='Suburban', y='Rating',
                                    title="Average Rating by City",
                                    labels={'Rating': 'Average Rating', 'Suburban': 'City'})
            st.plotly_chart(fig_rating_city, use_container_width=True)
            
            # Option 5: Property Type Distribution by Location
            st.subheader("Property Type Distribution by City")
            fig_property_city = px.histogram(df, x='Suburban', color='Property Type',
                                            title="Property Type Distribution by City",
                                            labels={'Property Type': 'Property Type', 'Suburban': 'City'})
            st.plotly_chart(fig_property_city, use_container_width=True)
            
            # Option 6: Geospatial Visualization (Map)
            st.subheader("Listing Distribution on Map")
            
            # Ensure you have Longitude and Latitude columns
            if 'Longitude' in df.columns and 'Latitude' in df.columns:
                fig_map = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", 
                                            hover_name="Name", hover_data=["Price", "Rating", "Property Type"],
                                            color_discrete_sequence=["fuchsia"], zoom=10, height=500)
                fig_map.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Geospatial data not available for mapping.")

            # Option 7: Price and Rating Correlation by City
            st.subheader("Correlation between Price and Rating by City")
            fig_corr_city = px.scatter(df, x='Price', y='Rating', color='Suburban',
                                    title="Correlation between Price and Rating by City",
                                    labels={'Price': 'Price (in currency)', 'Rating': 'Rating'})
            st.plotly_chart(fig_corr_city, use_container_width=True)

        

        # Call the Location Insights function within Streamlit app
        location_insights(df)

        # Function to create a GeoDataFrame
        def create_gdf(df):
            # Create the GeoDataFrame with geometry
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
            
            # Check if CRS is already set; if not, set it
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            
            # Transform to the desired CRS
            return gdf.to_crs(epsg=3857)

        # Step 1: Country selection
        # Convert the Country column to strings and drop NaN values
        country_options = sorted(df['Country'].dropna().astype(str).unique())
        country = st.selectbox('Select Country', country_options)

        # Filter data by selected country
        df_country = df[df['Country'].astype(str) == country]
        gdf_country = create_gdf(df_country)

        # Step 2: State selection
        if 'Market' in df_country.columns and not df_country['Market'].isna().all():
            state_options = sorted(df_country['Market'].dropna().astype(str).unique())
            state = st.selectbox('Select Market', state_options)

            # Filter data by selected state
            df_state = df_country[df_country['Market'].astype(str) == state]
            gdf_state = create_gdf(df_state)
        else:
            df_state = df_country
            gdf_state = gdf_country

        # Step 3: City selection
        if 'Suburban' in df_state.columns and not df_state['Suburban'].isna().all():
            city_options = sorted(df_state['Suburban'].dropna().astype(str).unique())
            city = st.selectbox('Select City', city_options)

            # Filter data by selected city
            df_city = df_state[df_state['Suburban'].astype(str) == city]
            gdf_city = create_gdf(df_city)
        else:
            df_city = df_state
            gdf_city = gdf_state

        # Step 4: Street selection
        if 'Government Area' in df_city.columns and not df_city['Government Area'].isna().all():
            street_options = sorted(df_city['Government Area'].dropna().astype(str).unique())
            Gv_area = st.selectbox('Select Area', street_options)

            # Filter data by selected street
            df_street = df_city[df_city['Government Area'].astype(str) == Gv_area]
            gdf_street = create_gdf(df_street)
        else:
            df_street = df_city
            gdf_street = gdf_city

        # Visualization based on the final selection
        fig = px.scatter_mapbox(
            gdf_street,
            lat="Latitude",
            lon="Longitude",
            hover_name="Government Area",
            hover_data=["Price", "Room Type", "Property Type", "Host Name"],
            color="Price",
            color_continuous_scale=px.colors.sequential.Viridis,
            size="No of Reviews",
            size_max=15,
            zoom=10 if gdf_street.shape[0] > 1 else 14,
            height=600
        )

        # Update map layout
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})

        # Display the map
    
        st.plotly_chart(fig)

        with st.container():
        # EDA 1 tab content

            # Ensure all filter columns are strings to avoid comparison errors
            df['Country'] = df['Country'].astype(str)
            df['Market'] = df['Market'].astype(str)
            df['Suburban'] = df['Suburban'].astype(str)

            # Initialize dynamic filters for specific columns
            dynamic_filters = DynamicFilters(df, filters=['Market', 'Country', 'Suburban'])

            # Sidebar for filters
            with st.sidebar:
                st.write("Apply filters in any order 👇")
                dynamic_filters.display_filters(location='sidebar')

                # Display filtered DataFrame
                dynamic_filters.display_df()

        with st.container():

            st.title("Airbnb Geospatial Visualization with Basic Details")
            
            # Create GeoDataFrame for filtered data
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
            gdf.set_crs(epsg=4326, inplace=True)
            gdf = gdf.to_crs(epsg=3857)

            # Price range filter
            price_filter = st.slider('Price Range', int(gdf['Price'].min()), int(gdf['Price'].max()), 
                                        (int(gdf['Price'].min()), int(gdf['Price'].max())))
            df_filtered = gdf[(gdf['Price'] >= price_filter[0]) & (gdf['Price'] <= price_filter[1])]

            # Center map based on filtered data
            center_lat = df_filtered['Latitude'].mean()
            center_lon = df_filtered['Longitude'].mean()

            # Create scatter mapbox
            fig = px.scatter_mapbox(
                df_filtered,
                lat="Latitude",
                lon="Longitude",
                hover_name="Country",
                hover_data=["Price", "Room Type", "Accomodates", "Host Name"],
                color="Price",
                color_continuous_scale=px.colors.sequential.Viridis,
                size="No of Reviews",
                size_max=15,
                zoom=10,
                center={"lat": center_lat, "lon": center_lon},
                height=500
            )
            fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig)

        with st.container():
        
            st.title('Visualization for Country, State, and City')

            # Create GeoDataFrame for entire dataset
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
            gdf.set_crs(epsg=4326, inplace=True)
            gdf = gdf.to_crs(epsg=3857)

            # Create scatter mapbox for entire dataset
            fig1 = px.scatter_mapbox(
                gdf,
                lat="Latitude",
                lon="Longitude",
                color="Price",
                size="Price",
                hover_name="Government Area",
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_data=['Suburban',"Market", "Country"],
                zoom=3,
                height=500
            )
            fig1.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig1)


    with tabs[3]:
        st.header("Interactive Map")



        # Function to handle the Interactive Tab
        def interactive_tab(df):
            st.title("Interactive Data Exploration")

            # 1. Advanced Filtering: Users can filter the data based on multiple conditions
            
            st.sidebar.header("Filter Options")
            
            # Filter by Price Range
            price_range = st.sidebar.slider("Select Price Range", int(df['Price'].min()), int(df['Price'].max()), (int(df['Price'].min()), int(df['Price'].max())))
            
            # Filter by Property Type
            property_types = df['Property Type'].unique()
            select_property_type = st.sidebar.multiselect("Select Property Type", property_types)
            
            # Filter by City
            cities = df['Suburban'].unique()
            select_city = st.sidebar.multiselect("Select City", cities)
            
            # Filter by Rating
            rating_range = st.sidebar.slider("Select Rating Range", 0, 100, (0, 100))
            
            # Apply filters to the DataFrame
            filtered_df = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]
            
            if select_property_type:
                filtered_df = filtered_df[filtered_df['Property Type'].isin(select_property_type)]
            
            if select_city:
                filtered_df = filtered_df[filtered_df['Suburban'].isin(select_city)]
            
            filtered_df = filtered_df[(df['Rating'] >= rating_range[0]) & (df['Rating'] <= rating_range[1])]

            # Display filtered data
            st.subheader("Filtered Data")
            st.dataframe(filtered_df)

            # Option to download the filtered data as CSV
            st.download_button(
                label="Download Filtered Data as CSV",
                data=filtered_df.to_csv().encode('utf-8'),
                file_name='filtered_data.csv',
                mime='text/csv'
            )

            # 2. Dynamic Data Exploration: Users can explore different metrics interactively
            
            st.subheader("Visualize the Data")
            
            # Select X and Y axis for the scatter plot
            x_axis = st.selectbox("Select X-axis for Scatter Plot", ['Price', 'Rating', 'No of Reviews', 'Availability_365','Accomodates'])
            y_axis = st.selectbox("Select Y-axis for Scatter Plot", ['Price', 'Rating', 'No of Reviews', 'Availability_365','Accomodates'])
            
            # Scatter plot
            st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color="Suburban", hover_data=['Name', 'Property Type', 'Rating'])
            st.plotly_chart(fig)

            # 3. Comparative Analysis: Users can compare multiple metrics with bar charts or box plots
            
            st.subheader("Compare Price by Property Type or City")
            
            # Select option for comparison
            compare_option = st.selectbox("Compare by", ['Property Type', 'Country'])
            
            # Generate a bar chart or box plot based on user selection
            if compare_option == 'Property Type':
                fig_compare = px.box(filtered_df, x='Property Type', y='Price', color='Property Type', title="Price Distribution by Property Type")
            elif compare_option == 'Country':
                fig_compare = px.box(filtered_df, x='Country', y='Price', color='Country', title="Price Distribution by Room Type")
            
            st.plotly_chart(fig_compare)

            # 4. Customizable Visualizations: Let users choose different chart types to visualize data
            
            st.subheader("Customizable Visualization")
            
            # Chart type selection
            chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Pie Chart", "Heatmap"])
            
            # Dynamic chart based on user selection
            if chart_type == "Line Chart":
                fig_line = px.line(filtered_df, x='Price', y='Rating', title="Price vs Rating")
                st.plotly_chart(fig_line)
            elif chart_type == "Bar Chart":
                fig_bar = px.bar(filtered_df, x='Property Type', y='Price', color='Suburban', title="Price by Property Type and City")
                st.plotly_chart(fig_bar)
            elif chart_type == "Pie Chart":
                fig_pie = px.pie(filtered_df, names='Property Type', values='Price', title="Price Distribution by Property Type")
                st.plotly_chart(fig_pie)
            elif chart_type == "Heatmap":
                corr_matrix = filtered_df[['Price', 'Rating', 'No of Reviews', 'Availability 365']].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                st.pyplot()

            # 5. Correlation Heatmap: Visualize correlations between various numerical features
            
            st.subheader("Correlation Heatmap")
            numeric_columns = ['Price', 'Rating', 'No of Reviews', 'Availability_365','Accomodates']
            corr_matrix = df[numeric_columns].corr()
            
            fig_corr = plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=1, linecolor='black')
            plt.title("Correlation Matrix")
            st.pyplot(fig_corr)



        # Call the function for interactive tab
        interactive_tab(df)

    

elif selected == "📞 Contact":
    st.title("Contact Us")
    st.write("Feel free to reach out!")
    st.write("If you have any queries or feedback, feel free to reach out via email.")

    # Add a form for user input (optional)
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send")
        
        if submitted:
            st.write("Thank you for reaching out! We'll get back to you soon.")

    # Add your Contact information here
    st.write("Email: gopiguru737@gmail.com")



