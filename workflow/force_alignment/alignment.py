import torch 
import numpy as np
import json  
import os 
from scipy.io.wavfile import write
from tqdm import tqdm 

from .asr_model import Model 
from .audio import load_audio,SAMPLE_RATE
from .utils import Point,Segment,parse_transcript_file
from huggingface_hub import snapshot_download,HfApi
import logging
import shutil


logger = logging.getLogger(__name__)


class ForceAligner:
    
    def __init__(self):
        self.model=self._load_model()
        self.output_path=".aligned_audios"
        #self.hf_token=hf_token
    
    def _load_model(self):
        
        if Model._instance is not None:
            return Model._instance
        else:
            return Model()
    
    def _extract_base(self,path):
        
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]

    def _compose_graph(self,emission,tokens,blank_id=0):
        """
        Compose a graph for force alignment using emission probabilities.

        Args:
            emission (torch.Tensor): Emission probabilities.
            tokens (list): List of token IDs.
            blank_id (int, optional): ID of the blank token. Defaults to 0.

        Returns:
            torch.Tensor: The composed graph.
        """

        num_frame=emission.size(0)
        num_tokens=len(tokens)

        graph=torch.zeros((num_frame,num_tokens))
        graph[1:,0]=torch.cumsum(emission[1:,blank_id],0)
        graph[0,1:]=-float("inf")
        graph[-num_tokens+1:,0]=float("inf")

        for t in range(num_frame-1):

            graph[t+1,1:]=torch.maximum(graph[t,1:]+emission[t,blank_id],
                                        graph[t,:-1]+emission[t,tokens[1:]],)
            
        return graph 
        
    def _backtrack(self,graph,emission,tokens,blank_id=0):
        """
        Backtrack to find the most probable path in the graph.

        Args:
        graph (torch.Tensor): Composed graph.
        emission (torch.Tensor): Emission probabilities.
        tokens (list): List of token IDs.
        blank_id (int, optional): ID of the blank token. Defaults to 0.

        Returns:
        list: The most probable paths, Point objects.
        """ 

        t,j=graph.size(0)-1,graph.size(1)-1

        path=[Point(j,t,emission[t,blank_id].exp().item())]
        while j>0:

            assert t>0

            p_stay=emission[t-1,blank_id]
            p_change=emission[t-1,tokens[j]]

            stayed=graph[t-1,j]+p_stay
            changed=graph[t-1,j-1]+p_change

            stayed=graph[t-1,j]+p_stay
            changed=graph[t-1,j-1]+p_change

            t-=1
            if changed>stayed:
                j -=1

            prob=(p_change if changed>stayed else p_stay).exp().item()
            path.append(Point(j,t,prob))

        while t>0:
            prob=emission[t-1,blank_id].exp().item()
            path.append(Point(j,t-1,prob))
            t-=1
        
        return path[::-1]
    
    def _merge_repeats(self,path,transcript):
        """
        Merge repeated segments in the path.

        Args:
        path (list): List of points in the path.
        transcript (str): The transcript corresponding to the path.

        Returns:
        list: Merged segments.
        """
        i1,i2=0,0
        segments=[]
        while i1<len(path):
            while i2<len(path) and path[i1].token_index == path[i2].token_index:
                i2+=1
            score=sum(path[k].score for k in range(i1,i2))/(i2-i1)

            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2-1].time_index+1,
                    score,
                )
            )
            i1=i2

        return segments
    def _merge_words(self,segments, separator="|"):
        """
        Merge word segments.

        Args:
        segments (list): List of segments.
        separator (str, optional): Separator for merging. Defaults to "|".

        Returns:
        list: Merged word segments.
        """
        
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words
    
    def _generate_audio_segments(self,wave_form,graph,word_segments,sample_rate):
        """
        Generate audio segments.

        Args:
        wave_form (numpy.ndarray): Audio waveform.
        graph (torch.Tensor): Composed graph.
        word_segments (list): List of word segments.
        sample_rate (int): Sampling rate of the audio.

        Returns:
        list: List of audio segments.
        """

        ratio=wave_form.shape[0]/graph.size(0)
        audio_segments=[]

        for i in range(len(word_segments)):
            word=word_segments[i]
            x0=int(ratio*word.start)
            x1=int(ratio*word.end)
            time_interval=f"{x0/ sample_rate:.3f}-{x1/sample_rate:.3f} sec"
            audio_seg=wave_form[x0:x1]
            audio_segments.append((word.label,time_interval,audio_seg))

        return audio_segments
    
    def _get_seg_duration(self,segments,sample_rate,audio,graph):

        ratio=audio.shape[0]/graph.size(0)
        dur_segment=[]
        for segment in segments:
            dict={}
            x0=int(ratio*segment.start)
            x1=int(ratio*segment.end)
            start_time=x0/sample_rate
            end_time=x1/sample_rate
            dict["start_dur"]=start_time
            dict["end_dur"]=end_time
            dict["audio_segment"]=audio[x0:x1]
            dur_segment.append((segment,dict))

        return dur_segment
    
    def _chunk_and_merge_segments(self,segments,chunk_duration):

        segment_chunks=[]
        current_chunk=[]
        audio_chunks=[]
        current_audio_segment=[]
        current_duration=0.0

        for segment,duration_info in segments:

            start_dur=duration_info['start_dur']
            end_dur=duration_info['end_dur']
            audio_seg=duration_info['audio_segment']
            duration=end_dur - start_dur

            if current_duration+duration > chunk_duration:
                segment_chunks.append(current_chunk)
                audio_chunks.append(np.concatenate(current_audio_segment))
                current_chunk=[]
                current_audio_segment=[]
                current_duration=0.0

            current_chunk.append(segment)
            current_audio_segment.append(audio_seg)
            current_duration+=duration

        if current_chunk:
            segment_chunks.append(current_chunk)
            audio_chunks.append(np.concatenate(current_audio_segment))

        return segment_chunks, audio_chunks
    
    def _merge_transcript(self,word_list):
        
        transcript=" "
        for word in word_list:
            transcript+=word.label
        return transcript
    
    def force_align(self,input_path:str,transcript:str,output_dir:str,alignment_duration=None):
        """
        Perform force alignment on the input audio file with the given transcript.

        Args:
        input_path (str): The path to the input audio file.
        transcript (str): The transcript to align with the audio.
        output_dir (str): The directory to save the output alignment files.

        Returns:
        None
        """
        #loading_audio
        audio=load_audio(input_path)
        token_ids=self.model.tokenize(transcript)
        preprocessed_transcript=transcript.replace(" ","|")

        #force_alignment
        emission=self.model.inference(audio)
        graph=self._compose_graph(emission,token_ids)
        path=self._backtrack(graph,emission,token_ids)
        segments =self._merge_repeats(path,preprocessed_transcript)

        #folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        folder_path=os.path.join(output_dir,self._extract_base(input_path))
        json_path=os.path.join(folder_path,"metadata.json")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if alignment_duration:
            
            dur_segment=self._get_seg_duration(segments,SAMPLE_RATE,audio,graph)
            segment_chunks,audio_segments=self._chunk_and_merge_segments(dur_segment,alignment_duration)

            ratio=audio.shape[0]/graph.size(0)
            transcripts=[]
            time_duration=[]
            for segment_chunk in segment_chunks:
                word_list=self._merge_words(segment_chunk)
                #print(word_list)
                transcripts.append(self._merge_transcript(word_list))
                x0=int(segment_chunk[0].start*ratio)
                x1=int(segment_chunk[-1].start*ratio)
                x0=x0/SAMPLE_RATE
                x1=x1/SAMPLE_RATE
                time_duration.append(f"{x0:.3f}-{x1:.3f} sec")
            
            json_dict={}
            segment_list=[]
            segment_paths=[]
            for i in range(len(audio_segments)):
                segment_paths.append(os.path.join(folder_path,f"segment_{i}.wav"))
            
            for i in range(len(segment_chunks)):
                segment_dict={}
                segment_dict["transcript"]=transcripts[i]
                segment_dict["duration"]=time_duration[i]
                segment_dict["file_path"]=segment_paths[i]
                segment_list.append(segment_dict)
                wave_form=audio_segments[i]
                write(segment_paths[i],8000,wave_form)
            
            json_dict["original_file_path"]=input_path
            json_dict["original_transcript"]=transcript
            json_dict["audio_segments"]=segment_list
        
        else:
            
            word_segments=self._merge_words(segments)

            #preparing audio segments
            json_dict={}
            segment_list=[]

            audio_segments=self._generate_audio_segments(audio,graph,word_segments,SAMPLE_RATE)
        
            segment_paths=[]
            for i in range(len(audio_segments)):
                segment_paths.append(os.path.join(folder_path,f"segment_{i}.wav"))
            
            
            for i in range(len(segment_paths)):
                segment_dict={}
                segment_dict["word_label"]=audio_segments[i][0]
                segment_dict["duration"]=audio_segments[i][1]
                segment_dict["file_path"]=segment_paths[i]
                segment_list.append(segment_dict)

                wave_form=audio_segments[i][2]
                write(segment_paths[i],8000,wave_form)
            
            json_dict["original_file_path"]=input_path
            json_dict["original_transcript"]=transcript
            json_dict["audio_segments"]=segment_list 

        with open(json_path,'w') as json_file:
            json.dump(json_dict,json_file,indent=4)

        #print(f"Force Alignment complete files save at {folder_path}")
    
    def load_dataset(self,dataset):
        
        path = snapshot_download(
            repo_id=dataset, repo_type="dataset", cache_dir="./data/"
        )
        return path
    
    def align_dataset(self,dataset,alignment_duration=None):

        path=self.load_dataset(dataset)
        
        file_list=os.listdir(path)
        
        if "transcription.txt" or "metadata.json" or "metadata.csv" in file_list:
            if "transcription.txt" in file_list:
                parsed_paths=parse_transcript_file(path,"transcription.txt")
                
                for i in tqdm(range(len(parsed_paths))):
                    input_file_path=parsed_paths[i][0]
                    transcript=parsed_paths[i][1]
                    self.force_align(input_file_path,transcript,self.output_path,alignment_duration)
            elif "metadata.json" in file_list:
                #TODO:add support for metdata.json
                pass 
            else:
                #TODO: add suport for metadata.csv
                pass 
        else:
            raise FileNotFoundError(f'Transcript file not found must contain file of type {file_list}')
        
        return self.output_path
            
    def push_to_hub(self,save_path,hf_token):
        
        logger.info("Pushing to hub")
        api = HfApi(endpoint="https://huggingface.co", token=hf_token)
        api.create_repo(repo_id=save_path, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=self.output_path,
            repo_id=save_path,
            repo_type="dataset",
        )
        logger.info("deleting data")
        shutil.rmtree(self.output_path)
        shutil.rmtree("./data/")
        
