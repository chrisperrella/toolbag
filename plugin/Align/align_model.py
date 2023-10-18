import mset

def get_selected_objects():
    selected_objects = mset.getSelectedObjects()
    
    selected_mesh_objects = [obj for obj in selected_objects if isinstance( obj, mset.MeshObject )]
    if len( selected_mesh_objects ) < 2:
        raise ValueError( "At least two mesh objects should be selected." )
    
    return selected_mesh_objects

def axis_to_index( axis ):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    return axis_map.get( axis, axis )

def align_to_value( axis, value ):
    selected_objects = [obj for obj in mset.getSelectedObjects() if isinstance( obj, mset.MeshObject )]

    if not selected_objects:
        print( "[Align Tool]: No MeshObject selected." )
        return

    for prim in selected_objects:
        new_position = prim.position.copy()
        new_position[axis] = value
        prim.position = new_position
        print( f'[Align Tool]: Aligning {prim.name} to {new_position}' )

def align(axis=0):
    selected_objects = get_selected_objects()
    target_prim_axis_value = selected_objects[0].position[axis]
    align_to_value(axis, target_prim_axis_value)

def align_to_min(axis=0):
    selected_objects = [obj for obj in mset.getSelectedObjects() if isinstance( obj, mset.MeshObject )]
    if not selected_objects:
        print( "[Align Tool]: No MeshObject selected." )
        return

    min_value = min(prim.position[axis] for prim in selected_objects)
    align_to_value( axis, min_value )

def align_to_max(axis=0):
    selected_objects = [obj for obj in mset.getSelectedObjects() if isinstance( obj, mset.MeshObject )]
    if not selected_objects:
        print( "[Align Tool]: No MeshObject selected." )
        return

    max_value = max( prim.position[axis] for prim in selected_objects )
    align_to_value( axis, max_value )

def distribute( axis = 0, 
                spacing =0.1 ):
    selected_objects = get_selected_objects()
    target_prim_axis_value = selected_objects[0].position[axis]

    for idx, prim in enumerate( selected_objects[1:] ):
        new_position = prim.position.copy()
        new_position[axis] = target_prim_axis_value + ( idx + 1 ) * spacing
        prim.position = new_position
        print( f'[Align Tool]: Distributing {prim.name} to {new_position}' )