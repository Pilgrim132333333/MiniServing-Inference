import Engine.Request
from Engine.Request import Request,RequestStatus
from Backend.BackendFactory import BackendFactory
from collections import deque
from Backend.ExecuteOutput import ExecuteOutput

class Scheduler:
    def __init__(self):
        self.WAITING_QUEUE = deque()
        self.RUNNING_QUEUE = deque()
        self.FINISHED_QUEUE = deque()
        self.FAILED_QUEUE = deque()
    
    def schedule(self):
        if not self.check_remaining():
            return None

        if len(self.RUNNING_QUEUE) > 0:
            request = self.RUNNING_QUEUE[0]
        else:
            request = self.WAITING_QUEUE[0]
            self.WAITING_QUEUE.popleft()
            request.set_status(RequestStatus.RUNNING)
            self.RUNNING_QUEUE.append(request)
        
        requests = [request]
        return requests

        
    def add_request(self,request:Request):
        #check status
        if request.get_status() == RequestStatus.WAITING:
            self.WAITING_QUEUE.append(request)
        else:
            raise ValueError("Request status is not WAITING")
        
    def check_remaining(self):
        if len(self.WAITING_QUEUE) > 0 or len(self.RUNNING_QUEUE) > 0:
            return True
        else:
            return False
    
    #Update the status of the request based on the result of executor
    def update(self,execute_output:ExecuteOutput = None):

        requests = execute_output.get_requests()
        past_key_values = execute_output.get_past_key_values()
        output_tokens = execute_output.get_output_tokens()
        for request in requests:
            status = request.get_status()

            if request.get_output_token() == []:
                request.set_output_token([output_tokens[requests.index(request)]])
            else:
                request.add_output_token([output_tokens[requests.index(request)]])

            if self._is_finished(request):
                request.set_status(RequestStatus.FINISHED)
                self.RUNNING_QUEUE.remove(request)
                self.FINISHED_QUEUE.append(request)
                self.FAILED_QUEUE.append(request)

            output_tokens = execute_output.get_output_tokens()
            output_token = output_tokens[requests.index(request)]
        if len(requests) == 1:
            past_key_value = execute_output.get_past_key_values()
            request.set_past_key_values(past_key_value)

        

    def _is_finished(self,request:Request):
        max_tokens = request.get_max_tokens()
        output_tokenIDs = request.get_output_token()
        curr_tokens = len(output_tokenIDs)
        if curr_tokens >= max_tokens or output_tokenIDs[-1] == request.get_eos_token_id():
            return True
        else:
            return False

        
        
        