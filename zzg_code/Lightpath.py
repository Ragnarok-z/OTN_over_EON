class Lightpath:
    def __init__(self, source, destination, transponder_mode, fs_allocated, path_in_G0):
        self.source = source
        self.destination = destination
        self.transponder_mode = transponder_mode
        self.fs_allocated = fs_allocated  # [began, end]
        self.path_in_G0 = path_in_G0  # Physical path in G0 layer
        self.capacity = transponder_mode["capacity"]
        self.used_capacity = 0
        self.demands = []  # List of demands using this lightpath

    def can_accommodate(self, demand):
        return (self.used_capacity + demand.traffic_class.value) <= self.capacity

    def remaining_capacity(self):
        return self.capacity - self.used_capacity

    def add_demand(self, demand):
        if self.can_accommodate(demand):
            self.demands.append(demand)
            self.used_capacity += demand.traffic_class.value
            return True
        return False

    def remove_demand(self, demand):
        if demand in self.demands:
            self.demands.remove(demand)
            self.used_capacity -= demand.traffic_class.value
            return True
        return False