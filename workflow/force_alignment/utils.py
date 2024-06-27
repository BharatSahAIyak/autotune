from dataclasses import dataclass 
import os 

@dataclass
class Point:
    token_index:int 
    time_index: int 
    score: float 

@dataclass
class Segment:
    label:str 
    start: int 
    end: int 
    score: float

    def __repr__(self) -> str:
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"
    
    @property
    def length(self):
        return self.end - self.start

def parse_transcript_file(base_path,transcript_file):
    
    file_path=os.path.join(base_path,transcript_file)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines.remove('\n')

    transcriptions = []
    for line in lines:

        strip_list=line.strip().split(' ', 1)
        audio_name, transcription =strip_list[0],strip_list[1]
        audio_name+=".wav"
        audio_path=os.path.join(base_path,audio_name)
        transcriptions.append((audio_path, transcription))

    return transcriptions
