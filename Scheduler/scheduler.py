import Engine.Request
from Engine.Request import Request,RequestStatus
from Backend.BackendFactory import BackendFactory
from collections import deque

class Scheduler:
    def __init__(self):
        self.WAITING_QUEUE = deque()
        self.RUNNING_QUEUE = deque()
        self.FINISHED_QUEUE = deque()
        self.FAILED_QUEUE = deque()
    
    def schedule(self):
        if not self.check_remaining():
            return None

        request = self.WAITING_QUEUE[0]
        self.WAITING_QUEUE.popleft()
        request.set_status(RequestStatus.RUNNING)
        self.RUNNING_QUEUE.append(request)
        
        return request

    def add_request(self,request:Request):
        #check status
        if request.get_status() == RequestStatus.WAITING:
            self.WAITING_QUEUE.append(request)
        else:
            raise ValueError("Request status is not WAITING")
        
    def check_remaining(self):
        if len(self.WAITING_QUEUE) > 0:
            return True
        else:
            return False
    
    def update(self):

        status = self.RUNNING_QUEUE[0].get_status()
        request = self.RUNNING_QUEUE[0]

        if status == RequestStatus.RUNNING: 
            self.RUNNING_QUEUE.popleft()
            self.FINISHED_QUEUE.append(request)
        elif status == RequestStatus.FINISHED:
            self.WAITING_QUEUE.popleft()
            self.FINISHED_QUEUE.append(request)
        elif status == RequestStatus.FAILED:
            self.WAITING_QUEUE.popleft()
            self.FAILED_QUEUE.append(request)
