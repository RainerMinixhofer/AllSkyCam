# Configuration file to produce an alt/az projection with the approximate zenit position of camera

# Default chart settings
DEFAULTS
plot_equator=0
plot_galactic_plane=0
plot_ecliptic=0
copyright=''
copyright_gap=0
copyright_gap_2=0
great_circle_key=0
magnitude_key=0
mag_min=5.0
ra_central=9.403361
dec_central=47.061743
position_angle=0
width=43.5
constellation_boundaries=0
constellation_sticks=1
ra_dec_lines=0
coords=ra_dec
star_names=1
star_label_mag_min=1.5

# Produce a copy of this chart using alt/az projection of the whole sky, with specified central point at the zenith
CHART
output_filename=allskycam.png
projection=alt_az
messier_col=0,0.8,0.8
galaxy_col=0,0,0
galaxy_col0=0,0,0
star_col=0.95,0.95,0.95
grid_col=0.3,0.3,0.3
equator_col=0.65,0,0.65
galactic_plane_col=0,0.8,0.25
ecliptic_col=0.8,0.65,0
constellation_label_col=0.4,0.4,0.4
