import ngsolve as ngs
from netgen.geom2d import CSG2d, Circle, Rectangle
from netgen.geom2d import EdgeInfo as EI, PointInfo as PI, Solid2d

def square(maxh : float = 0.1  # maximum element size
          ) -> ngs.Mesh : 
    """ unit-square mesh """
    return ngs.Mesh( ngs.unit_square.GenerateMesh(maxh = maxh) ) 


def gapedInductor(airgap : float = 1e-3, # if airgap put True
                  h : float = 0.005 # maximum element size in the air
                 ) -> ngs.Mesh:
    """ inductor core with a gap  """
    geo = CSG2d()
    # define some primitives
    box_ = Rectangle( pmin=(0.02,0.02), pmax=(0.08,0.08), mat="air", bc="out" )
    iron_ = Rectangle( pmin=(0.04,0.04), pmax=(0.06,0.06), mat="iron")
    airIron_ = Rectangle( pmin=(0.045,0.045), pmax=(0.055,0.055), mat="air")
    positiveCond_ = Rectangle( pmin=(0.05,0.045), pmax=(0.055,0.055), mat="condP")
    negativeCond_ = Rectangle( pmin=(0.06,0.045), pmax=(0.065,0.055), mat="condN")

    if airgap:
        airgap_ = Rectangle( pmin=(0.03,0.05 - airgap/2), pmax=(0.046,0.05 + airgap/2), mat="air")
        iron = iron_ - airIron_ - airgap_ - negativeCond_ - positiveCond_
        air = box_ - iron - positiveCond_ - negativeCond_ 
    else :
        iron = iron_ - airIron_ - negativeCond_ - positiveCond_
        air = box_  - iron - negativeCond_ - positiveCond_
        
    air.Mat("air")
    iron.Mat("iron").Maxh(h/10)
    geo.Add(air)
    geo.Add(iron)
    geo.Add(positiveCond_)
    geo.Add(negativeCond_)
    
    # generate mesh
    return ngs.Mesh(geo.GenerateMesh(maxh=h))

def capacitor( maxh : float = 0.1,   # maximum element size outside singularity
               maxh_singularity : float = 0.01 # maximum element size singularity
             ):
    
    geo = CSG2d()
    
    # define some primitives
    box = Rectangle( pmin=(0,0), pmax=(1,1), mat="air", right = "right", left = "left", bottom = "bottom", top = "top" )
    
    capacitor = Solid2d( [
        (0.3,0.4), PI(maxh=maxh_singularity), EI(bc="low"),
        (0.7,0.4), PI(maxh=maxh_singularity),
        (0.7,0.6), PI(maxh=maxh_singularity),  EI(bc="high"),
        (0.3,0.6), PI(maxh=maxh_singularity),
      ], mat="dielectric" )
    
    # add top level objects to geometry
    geo.Add(capacitor)
    geo.Add(box - capacitor)
    
    # generate mesh
    return ngs.Mesh(geo.GenerateMesh(maxh=maxh))
