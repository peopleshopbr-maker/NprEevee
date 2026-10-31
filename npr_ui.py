import bpy


def draw_main_buttons(layout):
    col = layout.column(align=True)
    col.operator("material.append_npr_nodes", text="Add NPR Shader")
    col.operator("material.append_outline", text="Add Outline")
    col.operator("npr.toon_camera", text="Setup Camera System")
    col.operator("npr.toon_lights", text="Add Toon Light")
    col.operator("npr.auto_rig", text="Auto Rig")


classes = ()
