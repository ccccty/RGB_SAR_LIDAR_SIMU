"""Generate a procedural quadcopter OBJ (600 mm class) for air-scene demos."""
import math
import os

V, F = [], []
def vert(x, y, z):
    V.append((x, y, z)); return len(V) - 1
def quad(a, b, c, d):
    F.append((a, b, c)); F.append((a, c, d))
def box(x0, x1, y0, y1, z0, z1):
    p = [vert(x0,y0,z0), vert(x1,y0,z0), vert(x1,y1,z0), vert(x0,y1,z0),
         vert(x0,y0,z1), vert(x1,y0,z1), vert(x1,y1,z1), vert(x0,y1,z1)]
    quad(p[0],p[3],p[2],p[1]); quad(p[4],p[5],p[6],p[7])
    quad(p[0],p[1],p[5],p[4]); quad(p[1],p[2],p[6],p[5])
    quad(p[2],p[3],p[7],p[6]); quad(p[3],p[0],p[4],p[7])
def cyl(cx, cy, z0, z1, r, n=14):
    bot, top = [], []
    for i in range(n):
        a = 2*math.pi*i/n
        bot.append(vert(cx + r*math.cos(a), cy + r*math.sin(a), z0))
        top.append(vert(cx + r*math.cos(a), cy + r*math.sin(a), z1))
    for i in range(n):
        j = (i+1) % n
        quad(bot[i], bot[j], top[j], top[i])
        F.append((bot[0], bot[j], bot[i]))
        F.append((top[0], top[i], top[j]))

# fuselage
box(-0.16, 0.16, -0.16, 0.16, 0.0, 0.10)
# arms (diagonal, from hub to motor pods)
for sx in (-1, 1):
    for sy in (-1, 1):
        # thin arm approximated by a rotated box built from its corner ring
        L, Wd = 0.28, 0.045
        dx, dy = sx * L / 2**0.5, sy * L / 2**0.5
        ux, uy = -sy * Wd / 2**0.5, sx * Wd / 2**0.5
        z0, z1 = 0.035, 0.065
        c = lambda a, b, z: vert(dx + a*ux + b*dx*0.0, dy + a*uy, z)
        pts = [c(-1,-1,z0), c(1,-1,z0), c(1,1,z0), c(-1,1,z0),
               c(-1,-1,z1), c(1,-1,z1), c(1,1,z1), c(-1,1,z1)]
        quad(pts[0],pts[3],pts[2],pts[1]); quad(pts[4],pts[5],pts[6],pts[7])
        quad(pts[0],pts[1],pts[5],pts[4]); quad(pts[1],pts[2],pts[6],pts[5])
        quad(pts[2],pts[3],pts[7],pts[6]); quad(pts[3],pts[0],pts[4],pts[7])
# motors + rotor discs
for sx in (-1, 1):
    for sy in (-1, 1):
        mx, my = sx*0.21, sy*0.21
        cyl(mx, my, 0.06, 0.105, 0.030)      # motor
        cyl(mx, my, 0.110, 0.118, 0.155)     # rotor disc (blurred)
# landing skids
for sy in (-1, 1):
    box(-0.06, 0.06, sy*0.10, sy*0.14, -0.05, 0.0)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "assets", "models", "uav_demo.obj")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("# simkit procedural quadcopter\n")
    for v in V:
        fh.write("v %.5f %.5f %.5f\n" % v)
    for f in F:
        fh.write("f %d %d %d\n" % (f[0]+1, f[1]+1, f[2]+1))
print("wrote", out, len(V), "verts", len(F), "tris")
