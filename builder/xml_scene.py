
import os
import lxml.etree as etree
import numpy as np

from dm_control import mjcf

class FlatScene:
    def __init__(self) -> None:
        self.mjcf_model = mjcf.RootElement(model="scenario")

        self.visual = self.mjcf_model.visual
        self.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        self.visual.headlight.ambient = [0.3, 0.3, 0.3]
        self.visual.headlight.specular = [0, 0, 0]

        self.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
        getattr(self.visual, "global").azimuth = 150
        getattr(self.visual, "global").elevation = -20
        
        self.mjcf_model.option.integrator = "implicitfast"
        self.mjcf_model.option.timestep = 0.0005
        # self.mjcf_model.option.sensornoise = "enable"

        self.asset = self.mjcf_model.asset
        self.asset.add(
            "texture",
            name="skybox",
            builtin="gradient",
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0, 0, 0],
            width=512,
            height=3072,
        )
        self.asset.add(
            "texture",
            type="2d",
            name="grouplane",
            builtin="checker",
            mark="edge",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            width=300,
            height=300,
            markrgb=[0.8, 0.8, 0.8],
        )
        self.asset.add(
            "material",
            name="groundplane_mat",
            texture="grouplane",
            texuniform="true",
            texrepeat=[5, 5],
            reflectance=0.2,
        )

        self.mjcf_model.worldbody.add(
            "light",
            name="global_light",
            pos=[0, 0, 3],
            dir=[0, 0, -1],
            directional="false",
        )
        self.mjcf_model.worldbody.add(
            "geom",
            name="floor",
            size=[0, 0, 0.125],
            type="plane",
            material="groundplane_mat",
            conaffinity=15,
            condim=3,
        )
        
    def save_xml(self, dir = ".", name = "scene.xml"):
        mjcf.export_with_assets(self.mjcf_model, dir, name)
        root = etree.parse(os.path.join(dir, name)).getroot()
        find_class = etree.XPath("//*[@class]")
        for c in find_class(root):
            c.attrib["class"] = "scene/"
        for num, bad in enumerate(root.xpath("//default[@class=\'robot/\']")):
            if num == 0:
                continue
            bad.getparent().remove(bad)
        with open(os.path.join(dir, name), "wb") as f:
            f.write(etree.tostring(root, pretty_print=True))
            
class Scene:
    def __init__(self) -> None:
        self.mjcf_model = mjcf.RootElement(model="scenario")

        self.visual = self.mjcf_model.visual
        self.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        self.visual.headlight.ambient = [0.3, 0.3, 0.3]
        self.visual.headlight.specular = [0, 0, 0]

        self.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
        getattr(self.visual, "global").azimuth = 150
        getattr(self.visual, "global").elevation = -20
        
        self.mjcf_model.option.integrator = "implicitfast"
        self.mjcf_model.option.timestep = 0.0005
        # self.mjcf_model.option.sensornoise = "enable"

        self.asset = self.mjcf_model.asset
        self.asset.add(
            "texture",
            name="skybox",
            builtin="gradient",
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0, 0, 0],
            width=512,
            height=3072,
        )
        self.asset.add(
            "texture",
            type="2d",
            name="grouplane",
            builtin="checker",
            mark="edge",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            width=300,
            height=300,
            markrgb=[0.8, 0.8, 0.8],
        )
        self.asset.add(
            "material",
            name="groundplane_mat",
            texture="grouplane",
            texuniform="true",
            texrepeat=[5, 5],
            reflectance=0.2,
        )

        self.mjcf_model.worldbody.add(
            "light",
            name="global_light",
            pos=[0, 0, 3],
            dir=[0, 0, -1],
            directional="false",
        )
        
    def save_xml(self, dir = ".", name = "scene.xml"):
        mjcf.export_with_assets(self.mjcf_model, dir, name)
        root = etree.parse(os.path.join(dir, name)).getroot()
        find_class = etree.XPath("//*[@class]")
        for c in find_class(root):
            c.attrib["class"] = "scene/"
        for num, bad in enumerate(root.xpath("//default[@class=\'robot/\']")):
            if num == 0:
                continue
            bad.getparent().remove(bad)
        with open(os.path.join(dir, name), "wb") as f:
            f.write(etree.tostring(root, pretty_print=True))
            
class SlideScene(FlatScene):
    def __init__(self, height = 0.150, angle = np.deg2rad(10)) -> None:
        super().__init__()
        self.height = height
        self.angle = angle
        self.high_ground = self.mjcf_model.worldbody.add("geom",
                                                        name="high_ground",
                                                        size=[2500/2, 1/2, height])
        self.mjcf_model.asset.add("mesh", name="slide", file="models/assets/slide10deg150h.stl")
        self.slide = self.mjcf_model.worldbody.add("geom",
                                                   pos=[21.01, 0, 0],
                                                   mesh="slide")