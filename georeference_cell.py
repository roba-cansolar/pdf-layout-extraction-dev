# Georeference Cell for full_pipeline.ipynb
# This cell georeferencesd the GeoJSON data by:
# 1. Swapping x,y coordinates 
# 2. Finding westernmost point with lowest y-coordinate
# 3. Translating based on reference coordinate (EPSG:4326) and site width

import json
import math
from shapely.geometry import shape, mapping
from shapely.ops import transform
import pyproj

def swap_coordinates(geom):
    """Swap x and y coordinates for all geometries"""
    def swap_coords(coords):
        if isinstance(coords[0], (list, tuple)):
            return [swap_coords(coord) for coord in coords]
        else:
            # Swap x,y coordinates
            return [coords[1], coords[0]]
    
    if geom['type'] == 'Point':
        geom['coordinates'] = swap_coords(geom['coordinates'])
    elif geom['type'] in ['LineString', 'MultiPoint']:
        geom['coordinates'] = swap_coords(geom['coordinates'])
    elif geom['type'] in ['Polygon', 'MultiLineString']:
        geom['coordinates'] = [swap_coords(ring) for ring in geom['coordinates']]
    elif geom['type'] == 'MultiPolygon':
        geom['coordinates'] = [[swap_coords(ring) for ring in polygon] for polygon in geom['coordinates']]
    elif geom['type'] == 'GeometryCollection':
        for geometry in geom['geometries']:
            swap_coordinates(geometry)
    
    return geom

def find_westernmost_lowest_point(geojson_data):
    """Find the westernmost point with the lowest y-coordinate from site_polygon features"""
    site_polygons = [feature for feature in geojson_data['features'] 
                    if feature['properties'].get('type') == 'site_polygon']
    
    if not site_polygons:
        raise ValueError("No site_polygon features found in GeoJSON")
    
    westernmost_x = float('inf')
    lowest_y_at_westernmost = float('inf')
    
    for feature in site_polygons:
        geom = shape(feature['geometry'])
        
        # Get all coordinates from the geometry
        if hasattr(geom, 'exterior'):
            coords = list(geom.exterior.coords)
        elif hasattr(geom, 'coords'):
            coords = list(geom.coords)
        else:
            # For multipolygons or complex geometries
            coords = []
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords.extend(list(poly.exterior.coords))
            elif geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
        
        for x, y in coords:
            # Check if this is more western (smaller x)
            if x < westernmost_x:
                westernmost_x = x
                lowest_y_at_westernmost = y
            elif x == westernmost_x and y < lowest_y_at_westernmost:
                lowest_y_at_westernmost = y
    
    return westernmost_x, lowest_y_at_westernmost

def calculate_site_width(geojson_data):
    """Calculate the east-west width of the site from site_polygon features"""
    site_polygons = [feature for feature in geojson_data['features'] 
                    if feature['properties'].get('type') == 'site_polygon']
    
    if not site_polygons:
        raise ValueError("No site_polygon features found in GeoJSON")
    
    min_x = float('inf')
    max_x = float('-inf')
    
    for feature in site_polygons:
        geom = shape(feature['geometry'])
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        min_x = min(min_x, bounds[0])
        max_x = max(max_x, bounds[2])
    
    return max_x - min_x

def georeference_geojson(geojson_data, reference_lat, reference_lon, actual_site_width_meters):
    """
    Georeference the GeoJSON data
    
    Parameters:
    - geojson_data: The GeoJSON data to georeference
    - reference_lat: Latitude of the westernmost lowest point (EPSG:4326)
    - reference_lon: Longitude of the westernmost lowest point (EPSG:4326) 
    - actual_site_width_meters: Actual width of the site in meters
    """
    
    # Step 1: Swap x,y coordinates for all features
    print("Step 1: Swapping x,y coordinates...")
    for feature in geojson_data['features']:
        swap_coordinates(feature['geometry'])
    
    # Step 2: Find the westernmost point with lowest y-coordinate
    print("Step 2: Finding reference point...")
    west_x, lowest_y = find_westernmost_lowest_point(geojson_data)
    print(f"Found reference point at: ({west_x:.6f}, {lowest_y:.6f})")
    
    # Step 3: Calculate current site width in coordinate units
    print("Step 3: Calculating site width...")
    current_site_width = calculate_site_width(geojson_data)
    print(f"Current site width: {current_site_width:.6f} coordinate units")
    
    # Step 4: Calculate scale factor from coordinate units to meters
    scale_meters_per_unit = actual_site_width_meters / current_site_width
    print(f"Scale: {scale_meters_per_unit:.6f} meters per coordinate unit")
    
    # Step 5: Set up coordinate transformation
    # We'll use a local UTM-like projection centered at the reference point
    # For simplicity, we'll use Web Mercator (EPSG:3857) as intermediate
    
    # Create transformer from geographic (lat/lon) to Web Mercator
    transformer_to_mercator = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_from_mercator = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    # Transform reference point to Web Mercator
    ref_mercator_x, ref_mercator_y = transformer_to_mercator.transform(reference_lon, reference_lat)
    print(f"Reference point in Web Mercator: ({ref_mercator_x:.2f}, {ref_mercator_y:.2f})")
    
    # Step 6: Calculate translation and scale
    # The reference point (west_x, lowest_y) should map to (ref_mercator_x, ref_mercator_y)
    
    def transform_coordinates(coords):
        """Transform coordinates from local coordinate system to geographic"""
        if isinstance(coords[0], (list, tuple)):
            return [transform_coordinates(coord) for coord in coords]
        else:
            x, y = coords
            
            # Scale from coordinate units to meters (relative to reference point)
            scaled_x = (x - west_x) * scale_meters_per_unit
            scaled_y = (y - lowest_y) * scale_meters_per_unit
            
            # Translate to Web Mercator coordinates
            mercator_x = ref_mercator_x + scaled_x
            mercator_y = ref_mercator_y + scaled_y
            
            # Transform back to geographic coordinates
            lon, lat = transformer_from_mercator.transform(mercator_x, mercator_y)
            
            return [lon, lat]
    
    # Step 7: Apply transformation to all features
    print("Step 6: Applying georeference transformation...")
    for feature in geojson_data['features']:
        geom = feature['geometry']
        
        if geom['type'] == 'Point':
            geom['coordinates'] = transform_coordinates(geom['coordinates'])
        elif geom['type'] in ['LineString', 'MultiPoint']:
            geom['coordinates'] = transform_coordinates(geom['coordinates'])
        elif geom['type'] in ['Polygon', 'MultiLineString']:
            geom['coordinates'] = [transform_coordinates(ring) for ring in geom['coordinates']]
        elif geom['type'] == 'MultiPolygon':
            geom['coordinates'] = [[transform_coordinates(ring) for ring in polygon] for polygon in geom['coordinates']]
        elif geom['type'] == 'GeometryCollection':
            for geometry in geom['geometries']:
                # Recursively transform each geometry in the collection
                temp_feature = {'geometry': geometry}
                transform_coordinates(temp_feature['geometry'])
    
    # Step 8: Add CRS information
    geojson_data['crs'] = {
        "type": "name",
        "properties": {
            "name": "EPSG:4326"
        }
    }
    
    print("Georeferencing complete!")
    return geojson_data

# Example usage (replace with your actual values):
# REFERENCE_LAT = 40.7128  # Replace with actual latitude
# REFERENCE_LON = -74.0060  # Replace with actual longitude  
# SITE_WIDTH_METERS = 1000  # Replace with actual site width in meters

# Assuming your GeoJSON data is in a variable called 'geojson_data'
# georeferenced_data = georeference_geojson(geojson_data, REFERENCE_LAT, REFERENCE_LON, SITE_WIDTH_METERS)

# To save the georeferenced data:
# with open('georeferenced_output.geojson', 'w') as f:
#     json.dump(georeferenced_data, f, indent=2)

print("Georeference functions defined. Set your reference coordinates and site width, then call:")
print("georeferenced_data = georeference_geojson(geojson_data, REFERENCE_LAT, REFERENCE_LON, SITE_WIDTH_METERS)")