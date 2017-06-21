import numpy as np

# Compute the relative position of the ego vehicle w.r.t. a reference vehicle
def bicycle_relative_integrate(delta_x_old, delta_y_old, delta_phi_old, wheel_angle_ego, wheel_angle_ref, speed_ego, speed_ref, length_front, length_rear, dt):
    
    # Compute derivatives
    (vx_ego, vy_ego, yr_ego) = bicycle_model(wheel_angle_ego, speed_ego, length_front, length_rear)
    (vx_ref, vy_ref, yr_ref) = bicycle_model(wheel_angle_ref, speed_ref, length_front, length_rear)
    
    # We will now integrate the speed of the two vehicles and obtain their position at t+1
    # in the reference frame of the OLD reference vehicle position
    
    # Change coordinates of derivatives for the ego vehicle
    vx_ego_ref = vx_ego * np.cos(delta_phi_old) - vy_ego * np.sin(delta_phi_old)
    vy_ego_ref = vx_ego * np.sin(delta_phi_old) + vy_ego * np.cos(delta_phi_old)
    
    # Integrate ego vehicle
    x_ego = delta_x_old + vx_ego_ref * dt
    y_ego = delta_y_old + vy_ego_ref * dt
    phi_ego = delta_phi_old + yr_ego * dt
    
    # Integrate reference vehicle
    x_ref = vx_ref * dt
    y_ref = vy_ref * dt
    phi_ref = yr_ref * dt
    
    # Obtain coordinates of ego vehicle in the reference frame of the NEW reference vehicle position
    delta_x = (x_ego - x_ref) * np.cos(phi_ref) + (y_ego - y_ref) * np.sin(phi_ref)
    delta_y = -(x_ego - x_ref) * np.sin(phi_ref) + (y_ego - y_ref) * np.cos(phi_ref)
    delta_phi = phi_ego - phi_ref
    
    return (delta_x, delta_y, delta_phi)

# Simple geometric bicycle model
def bicycle_model(wheel_angle, speed, length_front, length_rear):
    
    # Compute slip angle
    tan_slip_angle = length_rear * np.tan(wheel_angle) / (length_front + length_rear)
    slip_angle = np.arctan( tan_slip_angle )
    
    # Compute speed in relative coordinates
    vx = speed * np.cos(slip_angle)
    vy = speed * np.sin(slip_angle)
    yr = vx / (length_front + length_rear) * np.tan(wheel_angle)
    
    return (vx, vy, yr)
    