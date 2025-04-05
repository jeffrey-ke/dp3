

def main():
    for t in range(args.iter):
        rlbench = rlbench.init() # <-- important
        policy = dp3() # <-- important
        vggt = vggt() # <-- important
        cameras = rlbench.cameras
        images = cameras.capture()
        sensor = vggt.reconstruct(images) if args.use_vggt else rlbench.depth_camera.capture()
        policy.feed(sensor)
        rlbench.step()
        rlbench.visualize()

