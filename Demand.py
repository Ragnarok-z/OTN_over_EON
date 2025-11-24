from Tool import TrafficClass
import math

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
        return math.ceil(self.traffic_class.value / 10)


    def set_departure_time(self, holding_time):
        self.departure_time = self.arrival_time + holding_time