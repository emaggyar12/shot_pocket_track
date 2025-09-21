class EMABoxSmoother:
    def __init__(self, alpha_pos=0.4, alpha_size=0.3):
        self.alpha_pos = float(alpha_pos)
        self.alpha_size = float(alpha_size)
        self.state = None

    def update(self, bbox_xyxy):
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h = (x2 - x1), (y2 - y1)
        
        if self.state is None:
            self.state = (cx, cy, w, h)

        else:
            pcx, pcy, pw, ph = self.state
            cx = self.alpha_pos*cx + (1 - self.alpha_pos)*pcx
            cy = self.alpha_pos*cy + (1 - self.alpha_pos)*pcy
            w  = self.alpha_size*w + (1 - self.alpha_size)*pw
            h  = self.alpha_size*h + (1 - self.alpha_size)*ph
            self.state = (cx, cy, w, h)
        
        cx, cy, w, h = self.state
        x1s, y1s = cx - w*0.5, cy - h*0.5
        x2s, y2s = cx + w*0.5, cy + h*0.5
        return [int(x1s), int(y1s), int(x2s), int(y2s)]  # bounding boxes are required to be integers
    
    def mark_missed(self, decay=0.05):
        if self.state and decay:
            cx, cy, w, h, = self.state
            self.state = (cx, cy, (1-decay)*w, (1-decay)*h)