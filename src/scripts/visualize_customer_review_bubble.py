import pandas as pd
import numpy as np
import plotly.express as px
import requests

def get_cep_coordinates_improved(cep_prefixes):
    """Get coordinates for Brazilian postal code prefixes with better precision"""
    # First, try to use a CEP database that maps more precisely
    # If we don't have a full database, we can improve precision by using multiple digits
    
    coordinates = {}
    
    # More detailed mapping using first 2 digits for states/major regions
    # This gives us better resolution than just the first digit
    state_coords = {
        # São Paulo state
        '01': (-23.55, -46.63),  # São Paulo capital - central
        '02': (-23.51, -46.62),  # São Paulo - north zone
        '03': (-23.54, -46.57),  # São Paulo - east zone
        '04': (-23.62, -46.63),  # São Paulo - south zone
        '05': (-23.53, -46.69),  # São Paulo - west zone
        '06': (-23.53, -46.78),  # Osasco, Barueri region
        '07': (-23.46, -46.54),  # Guarulhos region
        '08': (-23.50, -46.41),  # East metropolitan area
        '09': (-23.67, -46.56),  # ABC Paulista region
        '11': (-23.96, -46.33),  # Santos, coastal SP
        '12': (-23.18, -45.88),  # São José dos Campos, Vale do Paraíba
        '13': (-22.90, -47.06),  # Campinas region
        '14': (-21.18, -47.81),  # Ribeirão Preto region
        '15': (-20.81, -49.38),  # São José do Rio Preto region
        '16': (-21.22, -50.42),  # Araçatuba region
        '17': (-22.33, -49.07),  # Bauru region
        '18': (-22.12, -51.38),  # Presidente Prudente region
        '19': (-22.73, -47.33),  # Piracicaba region
        
        # Rio de Janeiro state
        '20': (-22.90, -43.20),  # Rio de Janeiro capital - central
        '21': (-22.86, -43.24),  # Rio de Janeiro - north zone
        '22': (-22.97, -43.22),  # Rio de Janeiro - south zone
        '23': (-22.88, -43.31),  # Rio de Janeiro - west zone
        '24': (-22.88, -43.10),  # Niterói
        '25': (-22.66, -43.70),  # Petrópolis and mountain region
        '26': (-22.76, -43.44),  # Nova Iguaçu, Baixada Fluminense
        '27': (-22.51, -44.12),  # Interior of RJ state
        '28': (-21.76, -41.33),  # Campos, northern RJ
        
        # Minas Gerais
        '30': (-19.92, -43.94),  # Belo Horizonte - central
        '31': (-19.88, -43.92),  # Belo Horizonte - north
        '32': (-19.93, -43.98),  # Contagem
        '35': (-19.16, -42.91),  # Vale do Aço
        '36': (-21.76, -43.35),  # Juiz de Fora
        '37': (-20.73, -46.61),  # South of MG
        '38': (-18.91, -46.80),  # Northwest MG
        '39': (-17.86, -41.50),  # Vale do Jequitinhonha
        
        # South region
        '80': (-25.42, -49.27),  # Curitiba - central
        '81': (-25.50, -49.29),  # Curitiba - metropolitan
        '82': (-25.43, -49.27),  # Curitiba region
        '83': (-25.44, -49.27),  # Curitiba region
        '84': (-25.09, -50.16),  # Ponta Grossa
        '85': (-24.95, -53.46),  # Cascavel
        '86': (-23.31, -51.16),  # Londrina
        '87': (-23.42, -51.94),  # Maringá
        '88': (-27.59, -48.55),  # Florianópolis
        '89': (-26.92, -49.07),  # Blumenau, northeast SC
        '90': (-30.03, -51.22),  # Porto Alegre - central
        '91': (-30.02, -51.15),  # Porto Alegre - north
        '92': (-29.92, -51.18),  # Metropolitan region of POA
        '93': (-29.76, -51.14),  # Vale dos Sinos
        '95': (-29.16, -51.18),  # Caxias do Sul
        
        # Northeast region
        '40': (-12.97, -38.50),  # Salvador - central
        '41': (-12.94, -38.41),  # Salvador - northeast
        '42': (-12.41, -38.91),  # Region of Feira de Santana
        '49': (-10.91, -37.07),  # Aracaju
        '50': (-8.05, -34.92),   # Recife
        '51': (-8.04, -34.93),   # Recife metropolitan
        '52': (-8.03, -34.91),   # Recife region
        '53': (-8.28, -35.97),   # Caruaru
        '58': (-7.12, -34.86),   # João Pessoa
        '59': (-5.79, -35.21),   # Natal
        '60': (-3.73, -38.52),   # Fortaleza
        '64': (-5.09, -42.80),   # Teresina
        '65': (-2.53, -44.30),   # São Luís
        
        # North and Central West
        '66': (-1.46, -48.49),   # Belém
        '69': (-3.10, -60.02),   # Manaus
        '70': (-15.78, -47.93),  # Brasília - central
        '71': (-15.77, -47.76),  # Brasília - south
        '72': (-15.87, -48.01),  # Brasília - west
        '73': (-15.64, -47.80),  # Brasília - east
        '74': (-16.68, -49.26),  # Goiânia
        '75': (-16.33, -48.95),  # Anápolis
        '76': (-15.92, -50.13),  # West of Goiás
        '77': (-10.16, -48.33),  # Palmas, Tocantins
        '78': (-15.60, -56.10),  # Cuiabá
        '79': (-20.44, -54.65),  # Campo Grande
    }
    
    # For any 5-digit prefix, first try to match the first 2 digits
    for prefix in cep_prefixes:
        prefix_str = str(prefix)
        first_two = prefix_str[:2] if len(prefix_str) >= 2 else prefix_str
        
        # If we have a direct match for the first 2 digits, use that
        if first_two in state_coords:
            base_lat, base_lon = state_coords[first_two]
            
            # Add a tiny random offset to avoid all points in the same area clustering
            # This will spread points within the same prefix area for better visualization
            random_offset = (np.random.random() - 0.5) * 0.2  # ±0.1 degree offset
            coordinates[prefix] = (base_lat + random_offset, base_lon + random_offset)
        else:
            # Fallback to using the first digit (our original, less precise approach)
            region = prefix_str[0]
            region_coords = {
                '0': (-15.77, -47.92),  # Central
                '1': (-23.55, -46.63),  # São Paulo region
                '2': (-22.90, -43.20),  # Rio region
                '3': (-19.92, -43.94),  # Minas Gerais
                '4': (-25.42, -49.27),  # South
                '5': (-30.03, -51.22),  # Far South
                '6': (-16.68, -49.26),  # Central West
                '7': (-12.97, -38.50),  # Northeast
                '8': (-8.05, -34.92),   # North Northeast
                '9': (-3.10, -60.02)    # North
            }
            if region in region_coords:
                base_lat, base_lon = region_coords[region]
                random_offset = (np.random.random() - 0.5) * 0.3  # Larger offset
                coordinates[prefix] = (base_lat + random_offset, base_lon + random_offset)
            else:
                # Default to Brasília with random offset
                coordinates[prefix] = (-15.77 + (np.random.random() - 0.5) * 0.3, 
                                      -47.92 + (np.random.random() - 0.5) * 0.3)
    
    return coordinates

def create_brazil_postcode_map(df_postcodes):
    """Create an interactive map of Brazil showing customer satisfaction by postal code"""
    # Get coordinates for the postal codes
    cep_coords = get_cep_coordinates_improved(df_postcodes['customer_zip_code_prefix'])
    
    # Add coordinates to dataframe
    df_map = df_postcodes.copy()
    df_map['latitude'] = df_map['customer_zip_code_prefix'].map(lambda x: cep_coords[x][0])
    df_map['longitude'] = df_map['customer_zip_code_prefix'].map(lambda x: cep_coords[x][1])
    
    # Create interactive map with Plotly
    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        size="review_count",  # Bubble size based on review count
        color="avg_score",    # Color based on average score
        color_continuous_scale=px.colors.diverging.RdYlGn,  # Red for low scores, green for high
        range_color=[1, 5],   # Review scores range from 1-5
        size_max=40,          # Maximum bubble size 
        zoom=4,               # Zoom level
        mapbox_style="carto-positron",  # Map style
        hover_name="customer_zip_code_prefix",
        hover_data=["review_count", "avg_score"],
        title="Customer Satisfaction by Region in Brazil"
    )
    
    # Add state borders to the map
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        coloraxis_colorbar=dict(
            title="Avg Review Score",
            tickvals=[1, 2, 3, 4, 5],
        ),
        mapbox=dict(
            center=dict(lat=-15.77, lon=-47.92),  # Center on Brazil
            zoom=4
        )
    )
    
    return fig