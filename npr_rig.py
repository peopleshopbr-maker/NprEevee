"""Automatic rigging for biped and quadruped type meshes."""

import math

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

from .npr_common import UP, front_vector, subject_meshes, world_points


class Subject:
    def __init__(self, points, front):
        self.front = front.normalized()
        self.left = UP.cross(self.front).normalized()
        P = points
        l = np.array(self.left)
        f = np.array(self.front)
        self.u = P @ l
        self.v = P @ f
        self.w = P[:, 2]
        self.umin, self.umax = float(self.u.min()), float(self.u.max())
        self.vmin, self.vmax = float(self.v.min()), float(self.v.max())
        self.zmin, self.zmax = float(self.w.min()), float(self.w.max())
        self.H = max(self.zmax - self.zmin, 1e-5)
        self.uc = (self.umin + self.umax) / 2.0
        self.vc = (self.vmin + self.vmax) / 2.0
        self.halfwidth = max(self.umax - self.uc, 1e-5)
        self.length = max(self.vmax - self.vmin, 1e-5)

    def pt(self, u, v, w):
        return self.left * u + self.front * v + Vector((0.0, 0.0, w))

    def Z(self, f):
        return self.zmin + self.H * f


def detect_rig_type(points, front):
    zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
    height = zmax - zmin
    if height <= 0:
        return 'HUMANOID'
    feet = points[points[:, 2] < zmin + 0.15 * height]
    if len(feet) < 4:
        return 'HUMANOID'
    foot_span = max(float(np.ptp(feet[:, 0])), float(np.ptp(feet[:, 1])))
    body_span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    return 'DOG' if (foot_span > 0.45 * height and body_span > 0.8 * height) else 'HUMANOID'


def humanoid_layout(subject, pose):
    H = subject.H
    Z = subject.Z
    pt = subject.pt
    uc, vc = subject.uc, subject.vc
    specs = []

    def add(name, head, tail, parent=None, connect=False):
        specs.append((name, head, tail, parent, connect))

    ratio = subject.halfwidth / H
    if pose == 'AUTO':
        pose = 'T' if ratio > 0.42 else ('A' if ratio > 0.24 else 'REST')

    add("root", pt(uc, vc, subject.zmin), pt(uc, vc + 0.12 * H, subject.zmin))
    add("hips", pt(uc, vc, Z(0.50)), pt(uc, vc, Z(0.57)), "root")
    add("spine", pt(uc, vc, Z(0.57)), pt(uc, vc, Z(0.67)), "hips", True)
    add("chest", pt(uc, vc, Z(0.67)), pt(uc, vc, Z(0.80)), "spine", True)
    add("neck", pt(uc, vc, Z(0.80)), pt(uc, vc, Z(0.86)), "chest", True)
    add("head", pt(uc, vc, Z(0.86)), pt(uc, vc, subject.zmax), "neck", True)

    shoulder_off = 0.10 * H
    shoulder_height = Z(0.795)
    for side, sign in ((".L", 1.0), (".R", -1.0)):
        shoulder = pt(uc + sign * shoulder_off, vc, shoulder_height)
        add("clavicle" + side, pt(uc + sign * 0.012 * H, vc, Z(0.795)), shoulder, "chest")
        if pose == 'T':
            d = Vector((1.0, 0.0))
            lengths = (0.45 * (subject.halfwidth), 0.35 * (subject.halfwidth), 0.20 * (subject.halfwidth))
        else:
            ang = math.radians(35.0 if pose == 'A' else 10.0)
            d = Vector((math.sin(ang), -math.cos(ang)))
            lengths = (0.17 * H, 0.15 * H, 0.08 * H)

        def at(dist):
            return pt(uc + sign * (shoulder_off + d.x * dist), vc, shoulder_height + d.y * dist)

        elbow, wrist, hand_tip = at(lengths[0]), at(lengths[0] + lengths[1]), at(sum(lengths))
        add("upper_arm" + side, shoulder, elbow, "clavicle" + side, True)
        add("forearm" + side, elbow, wrist, "upper_arm" + side, True)
        add("hand" + side, wrist, hand_tip, "forearm" + side, True)

        thigh_x = uc + sign * 0.055 * H
        ankle_v = vc - 0.02 * H
        add("thigh" + side, pt(thigh_x, vc, Z(0.50)), pt(thigh_x, vc, Z(0.28)), "hips")
        add("shin" + side, pt(thigh_x, vc, Z(0.28)), pt(thigh_x, ankle_v, Z(0.05)), "thigh" + side, True)
        foot_ball = pt(thigh_x, ankle_v + 0.09 * H, Z(0.015))
        add("foot" + side, pt(thigh_x, ankle_v, Z(0.05)), foot_ball, "shin" + side, True)
        add("toe" + side, foot_ball, pt(thigh_x, ankle_v + min(0.15 * H, max(subject.length / 2.0, 0.05 * H)) + 0.03 * H, Z(0.01)), "foot" + side, True)

    return specs


def dog_layout(subject):
    specs = []

    def add(name, head, tail, parent=None, connect=False):
        specs.append((name, head, tail, parent, connect))

    def V(t):
        return subject.vmin + subject.length * t

    def Z(f):
        return subject.Z(f)

    def P(u, t, f):
        return subject.pt(u, V(t), Z(f))

    c = subject.uc
    add("root", P(c, 0.50, 0.0), P(c, 0.56, 0.0))
    add("hips", P(c, 0.22, 0.66), P(c, 0.36, 0.68), "root")
    add("spine", P(c, 0.36, 0.68), P(c, 0.50, 0.70), "hips", True)
    add("chest", P(c, 0.50, 0.70), P(c, 0.64, 0.71), "spine", True)
    add("neck", P(c, 0.64, 0.71), P(c, 0.80, 0.84), "chest", True)
    add("head", P(c, 0.80, 0.84), P(c, 0.92, 0.86), "neck", True)
    add("snout", P(c, 0.92, 0.86), P(c, 1.00, 0.78), "head", True)
    add("tail.001", P(c, 0.22, 0.66), P(c, 0.12, 0.74), "hips")
    add("tail.002", P(c, 0.12, 0.74), P(c, 0.06, 0.80), "tail.001", True)
    add("tail.003", P(c, 0.06, 0.80), P(c, 0.00, 0.82), "tail.002", True)

    leg_width = min(0.5 * subject.halfwidth, 0.35 * subject.length)
    for side, sign in ((".L", 1.0), (".R", -1.0)):
        u = c + sign * leg_width
        add("front_clavicle" + side, P(c + sign * 0.1 * subject.halfwidth, 0.60, 0.68), P(u, 0.60, 0.60), "chest")
        add("front_upper" + side, P(u, 0.60, 0.60), P(u, 0.57, 0.34), "front_clavicle" + side, True)
        add("front_lower" + side, P(u, 0.57, 0.34), P(u, 0.59, 0.10), "front_upper" + side, True)
        add("front_paw" + side, P(u, 0.59, 0.10), P(u, 0.65, 0.015), "front_lower" + side, True)
        add("rear_thigh" + side, P(u, 0.22, 0.62), P(u, 0.29, 0.36), "hips")
        add("rear_shin" + side, P(u, 0.29, 0.36), P(u, 0.20, 0.18), "rear_thigh" + side, True)
        add("rear_foot" + side, P(u, 0.20, 0.18), P(u, 0.21, 0.04), "rear_shin" + side, True)
        add("rear_paw" + side, P(u, 0.21, 0.04), P(u, 0.26, 0.015), "rear_foot" + side, True)

    return specs


def build_armature(context, name, specs, collection):
    armature_data = bpy.data.armatures.new(name)
    armature_obj = bpy.data.objects.new(name, armature_data)
    collection.objects.link(armature_obj)
    armature_data.display_type = 'OCTAHEDRAL'
    armature_obj.show_in_front = True

    for obj in context.view_layer.objects:
        obj.select_set(False)
    armature_obj.select_set(True)
    context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        edit_bones = armature_data.edit_bones
        for bname, head, tail, parent, connect in specs:
            bone = edit_bones.new(bname)
            bone.head = head
            bone.tail = tail
            if (bone.tail - bone.head).length < 1e-4:
                bone.tail = bone.head + Vector((0.0, 0.0, 1e-3))
            if parent:
                bone.parent = edit_bones[parent]
                bone.use_connect = connect
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj


def bind_meshes(context, armature_obj, meshes):
    world = {m.name: m.matrix_world.copy() for m in meshes}
    for mesh in meshes:
        mesh.parent = None
        mesh.matrix_world = world[mesh.name]

    for mesh in meshes:
        if mesh.data.users > 1:
            mesh.data = mesh.data.copy()

    for obj in context.view_layer.objects:
        obj.select_set(False)
    for mesh in meshes:
        mesh.select_set(True)
    context.view_layer.objects.active = meshes[0]
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except RuntimeError:
        pass

    for obj in context.view_layer.objects:
        obj.select_set(False)
    for mesh in meshes:
        mesh.select_set(True)
    armature_obj.select_set(True)
    context.view_layer.objects.active = armature_obj
    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO', keep_transform=True)
    except RuntimeError:
        bpy.ops.object.parent_set(type='ARMATURE', keep_transform=True)

    return "armature"


class NPR_OT_auto_rig(Operator):
    bl_idname = "npr.auto_rig"
    bl_label = "Auto Rig"
    bl_description = "Create a simple auto-rig and bind it to the selected character or dog mesh."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cfg = context.scene.npr_tools
        meshes = subject_meshes(context, fallback_all=True)
        if not meshes:
            self.report({'ERROR'}, "Select a mesh before applying an auto-rig")
            return {'CANCELLED'}

        points = world_points(meshes)
        if points is None or len(points) < 4:
            self.report({'ERROR'}, "Selected mesh does not contain valid geometry")
            return {'CANCELLED'}

        front = front_vector(cfg)
        rig_kind = cfg.rig_type
        if rig_kind == 'AUTO':
            rig_kind = detect_rig_type(points, front)

        subject = Subject(points, front)
        if rig_kind == 'DOG':
            specs = dog_layout(subject)
            name = "NPR Dog Rig"
        else:
            specs = humanoid_layout(subject, cfg.rig_pose)
            name = "NPR Character Rig"

        collection = bpy.data.collections.get("NPR Rig")
        if collection is None:
            collection = bpy.data.collections.new("NPR Rig")
            context.scene.collection.children.link(collection)

        armature_obj = build_armature(context, name, specs, collection)
        bind_meshes(context, armature_obj, meshes)
        self.report({'INFO'}, f"Created {name} and bound the selected mesh")
        return {'FINISHED'}


classes = (NPR_OT_auto_rig,)
