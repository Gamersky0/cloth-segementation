class DetectEdgeResponse:
    # DetectEdgeResponse 包含以下成员变量
    def __init__(self):
        self.rgb_im = None
        self.depth_im = None
        self.prediction = None
        self.corners = None
        self.outer_edges = None
        self.inner_edges = None