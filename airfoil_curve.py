# SpaceClaim Script — NACA 2412 Airfoil
# API: V252
# Run via: Scripting tab -> Run Script

coords = [
    (600.0000, 0.1300),
    (595.0000, 1.1400),
    (590.0000, 2.0800),
    (580.0000, 3.7500),
    (570.0000, 5.1800),
    (560.0000, 6.3600),
    (550.0000, 7.2400),
    (540.0000, 7.8000),
    (530.0000, 7.8800),
    (525.0000, 7.6700),
    (520.0000, 7.2600),
    (515.0000, 6.6100),
    (510.0000, 5.6300),
    (507.5000, 4.9600),
    (505.0000, 4.1300),
    (502.5000, 2.9900),
    (501.2500, 2.1500),
    (500.0000, 0.0000),
    (501.2500, -1.6500),
    (502.5000, -2.2700),
    (505.0000, -3.0100),
    (507.5000, -3.4600),
    (510.0000, -3.7500),
    (515.0000, -4.1000),
    (520.0000, -4.2300),
    (525.0000, -4.2200),
    (530.0000, -4.1200),
    (540.0000, -3.8000),
    (550.0000, -3.3400),
    (560.0000, -2.7600),
    (570.0000, -2.1400),
    (580.0000, -1.5000),
    (590.0000, -0.8200),
    (595.0000, -0.4800),
    (600.0000, -0.1300),
]

# Build SpaceClaim Point objects from coordinates
pts = [Point.Create(MM(x), MM(y), MM(0)) for x, y in coords]

# Draw line segments connecting each coordinate point
# This creates a polyline approximation of the airfoil
for i in range(len(pts) - 1):
    segment = Line.CreateBound(pts[i], pts[i+1])
    DesignCurve.Create(GetRootPart(), segment)

# Close the airfoil by connecting last point back to first
segment = Line.CreateBound(pts[-1], pts[0])
DesignCurve.Create(GetRootPart(), segment)

print('NACA 2412 airfoil drawn successfully!')
print('Total points: ' + str(len(pts)))