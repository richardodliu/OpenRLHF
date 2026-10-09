"""Exact enumeration of the revised two-policy paper examples (no training)."""
from fractions import Fraction as F
from itertools import product
import json

paths = list(product(range(3), repeat=2))

def evaluate(pfirst, qfirst, qsecond, reward):
    pm = {y: pfirst[y[0]] / 3 for y in paths}
    qs = lambda i: qsecond.get(i, [F(1, 3)] * 3)
    ratio = {y: [qfirst[y[0]] / pfirst[y[0]], 3 * qs(y[0])[y[1]]] for y in paths}
    mask = {y: [F(4, 5) <= ratio[y][0] <= F(5, 4),
                F(4, 5)**2 <= ratio[y][0] * ratio[y][1] <= F(5, 4)**2] for y in paths}
    S = D = F(0)
    for y, z in product(paths, repeat=2):
        A = reward(y) - reward(z)
        weight = pm[y] * pm[z]
        signal = {y: A, z: -A} if y != z else {y: F(0)}
        # H=A and lambda=1; common scale cancels in the margin.
        S += weight * sum(signal[v] * sum(m*r for m, r in zip(mask[v], ratio[v])) for v in (y,z)) / 4
        D += weight * sum(abs(signal[v]) * sum((1-m)*r for m, r in zip(mask[v], ratio[v])) for v in (y,z)) / 4
    gain = sum(qfirst[y[0]] * qs(y[0])[y[1]] * reward(y) - pm[y]*reward(y) for y in paths)
    moments = [sum(pm[y]*mask[y][t]*ratio[y][t]**2 for y in paths) for t in range(2)]
    return S, D, gain, moments, mask

checks=[]
for B in [F(8,5), F(2), F(10), F(1000)]:
    zeta=F(1,100);a=1-zeta;delta=F(1,30)
    result=evaluate([a,zeta/(B+1),zeta*B/(B+1)],
                    [a,zeta*B/(B+1),zeta/(B+1)],
                    {0:[F(1,3)+delta,F(1,3)-delta,F(1,3)]},
                    lambda y:F(y==(0,0)))
    S,D,gain,moments,mask=result
    assert S==a*zeta/3+a*delta/2 and D==a*zeta/3
    assert 2*(S-D)==gain==a*delta
    assert moments==[a,a*(1+6*delta**2)]
    assert all(m==[y[0]==0,y[0]==0] for y,m in mask.items())
    assert 2*(S-D)-4*zeta*delta==F(19,600)
    checks.append({'example':'positive','B':str(B),'margin':str(2*(S-D)),'lower_bound':'19/600'})
p=[F(1,2),F(1,6),F(1,3)];q=[F(1,2),F(1,3),F(1,6)]
base=evaluate(p,q,{},lambda y:F(y[1]==0))
for delta in [F(1,1000),F(1,90),F(1,45)]:
    result=evaluate(p,q,{0:[F(1,3)+delta,F(1,3)-delta,F(1,3)],
                         1:[F(1,3)-3*delta,F(1,3)+3*delta,F(1,3)]},lambda y:F(y[1]==0))
    assert result[0]-base[0]==delta/4
    assert result[2]-base[2]==-delta/2
    assert all(m==[y[0]==0,y[0]==0] for y,m in result[4].items())
    checks.append({'example':'shared_parameter','delta':str(delta),'objective_increment':str(delta/4),'reward_increment':str(-delta/2)})
print(json.dumps({'status':'passed','arithmetic':'exact rational','checks':checks},indent=2))
