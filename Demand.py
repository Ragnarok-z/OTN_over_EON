from Tool import TrafficClass

class Demand:
    def __init__(self, id, source, destination, traffic_class, arrival_time):
        self.id = id
        self.source = source
        self.destination = destination
        self.traffic_class = traffic_class
        self.arrival_time = arrival_time
        self.departure_time = None
        self.path = None
        self.lightpaths_used = []
        self.required_slots = self.calculate_required_slots()

    def calculate_required_slots(self):
        # Calculate required OTN timeslots based on traffic class
        if self.traffic_class == TrafficClass.GE10:
            return 1  # Simplified - actual calculation would be based on OTN timeslot size
        elif self.traffic_class == TrafficClass.GE100:
            return 10
        elif self.traffic_class == TrafficClass.GE400:
            return 40
        return 1

    def set_departure_time(self, holding_time):
        self.departure_time = self.arrival_time + holding_time