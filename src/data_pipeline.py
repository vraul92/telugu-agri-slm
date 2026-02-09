"""
Data Pipeline for Telugu Agricultural SLM
Collects and prepares Telugu agricultural text corpus
"""
import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Iterator, Optional
import requests
from tqdm import tqdm


class TeluguAgriDataCollector:
    """
    Collects and prepares Telugu agricultural text data.
    Generates synthetic data for proof of concept.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Telugu agricultural vocabulary
        self.crops = [
            "వరి", "మొక్కజొన్న", "పత్తి", "గోధుమ", "సజ్జ", "రాగి", "జొన్న",
            "పప్పు దినుసులు", "నువ్వులు", "పచ్చిమిరప", "బీన్స్", "బఠానీలు",
            "టమాటో", "బంగాళాదుంప", "వంకాయ", "దోసకాయ", "పుచ్చకాయ", "కర్బూజ",
        ]
        
        self.pests = [
            "తెగులు", "పురుగు", "వంకడ పురుగు", "ఆకు తెగులు", "కాండం తెగులు",
            "పండ్ల తెగులు", "వేరు కుళ్ళు", "ఈగ", "పచ్చదిద్ద", "తెలుడు తెగులు",
        ]
        
        self.fertilizers = [
            "యూరియా", "డీఏపీ", "ఎంఓపీ", "ఎస్ఎస్పీ", "జింక్ సల్ఫేట్",
            "బోరాన్", "ఇనుము", "సేంద్రీయ ఎరువు", "వర్మి కంపోస్ట్",
        ]
        
        self.soil_types = [
            "నల్లరేగడి", "ఎర్ర", "ఇసుక", "లోయ",
        ]
        
        self.seasons = [
            "ఖరీఫ్", "రబీ", "వేసవి", "వర్షాకాలం", "శీతాకాలం",
        ]
        
        self.irrigation_types = [
            "చుక్కల నీటిపారుదల", "తెల్లటి నీటిపారుదల", "ట్రికిల్ ఇరిగేషన్",
            "స్ప్రింక్లర్", "వర్షాధార",
        ]
        
    def generate_qa_pairs(self, num_samples: int = 5000) -> List[Dict]:
        """Generate synthetic Q&A pairs in Telugu"""
        qa_templates = [
            {
                "question": "{crop} పంట ఎప్పుడు వేయాలి?",
                "answer": "{crop} పంటను {season} ఋతువులో వేయడం మంచిది. {soil} నేలలో బాగా పెరుగుతుంది."
            },
            {
                "question": "{crop} పంటకు ఏమైనా {pest} వస్తే ఏమి చేయాలి?",
                "answer": "{crop} పంటకు {pest} వస్తే, వెంటనే {fertilizer} స్ప్రే చేయండి. దీనివల్ల తెగులు నియంత్రణలో ఉంటుంది."
            },
            {
                "question": "{crop} పంటకు ఎంత నీరు అవసరం?",
                "answer": "{crop} పంటకు {irrigation} పద్ధతిలో 7-10 రోజులకు ఒకసారి నీరు పెట్టాలి. ఎక్కువ నీరు పండుకు హానికరం."
            },
            {
                "question": "{crop} పంటకు ఎన్ని రకాల {fertilizer} వేయాలి?",
                "answer": "{crop} పంటకు నేల పరీక్ష ఆధారంగా {fertilizer} మరియు ఇతర సూక్ష్మ పోషకాలు వేయాలి. అధిక మోతాదులో ఎరువులు వేయకూడదు."
            },
            {
                "question": "{crop} పంట ధర ఎంత ఉంటుంది?",
                "answer": "{crop} పంట ధర మార్కెట్ పరిస్థితులను బట్టి మారుతూ ఉంటుంది. రైతులకు మద్దతు ధర కల్పించబడుతుంది."
            },
            {
                "question": "{soil} నేలలో ఏ పంటలు పండిస్తారు?",
                "answer": "{soil} నేలలో {crop} మరియు {crop2} పంటలు బాగా పండుతాయి. దీనికి {irrigation} అవసరం."
            },
        ]
        
        qa_pairs = []
        for i in range(num_samples):
            template = qa_templates[i % len(qa_templates)]
            
            # Fill in placeholders
            qa = {
                "question": template["question"].format(
                    crop=random.choice(self.crops),
                    pest=random.choice(self.pests),
                    season=random.choice(self.seasons),
                    soil=random.choice(self.soil_types),
                    fertilizer=random.choice(self.fertilizers),
                    irrigation=random.choice(self.irrigation_types),
                    crop2=random.choice(self.crops),
                ),
                "answer": template["answer"].format(
                    crop=random.choice(self.crops),
                    pest=random.choice(self.pests),
                    season=random.choice(self.seasons),
                    soil=random.choice(self.soil_types),
                    fertilizer=random.choice(self.fertilizers),
                    irrigation=random.choice(self.irrigation_types),
                    crop2=random.choice(self.crops),
                ),
            }
            qa_pairs.append(qa)
        
        return qa_pairs
    
    def generate_agricultural_text(self, num_samples: int = 10000) -> List[str]:
        """Generate synthetic agricultural documents in Telugu"""
        
        document_templates = [
            "{crop} పంట సాగు:\n{crop} ఒక ముఖ్యమైన పంట. దీనిని {season} ఋతువులో సాగు చేస్తారు. "
            "{soil} నేలలో ఇది బాగా పెరుగుతుంది. {irrigation} విధానంలో నీటి నిర్వహణ చేయాలి. "
            "{fertilizer} వంటి ఎరువులను సమయానికి వేయాలి. {pest} వంటి తెగుళ్ళను జాగ్రత్తగా చూసుకోవాలి.",
            
            "వ్యవసాయ సలహా:\nరైతులు {crop} పంట పండించేటప్పుడు క్రింది విషయాలు గుర్తుంచుకోవాలి:\n"
            "1. మంచి రకం విత్తనాలను ఎంచుకోండి\n"
            "2. {soil} నేల సారాన్ని పరీక్షించుకోండి\n"
            "3. {irrigation} పద్ధతిని అనుసరించండి\n"
            "4. {fertilizer} ఎరువులను సరైన మోతాదులో వేయండి\n"
            "5. {pest} తెగులు నివారణకు చర్యలు తీసుకోండి",
            
            "నేల సంరక్షణ:\n{soil} నేలలో {crop} సాగు చేయడానికి ప్రత్యేక జాగ్రత్తలు అవసరం. "
            "సేంద్రీయ ఎరువుల వాడకం వల్ల నేల సారం పెరుగుతుంది. "
            "నీటి నిర్వహణ {irrigation} పద్ధతిలో ఉండాలి.",
        ]
        
        texts = []
        for i in range(num_samples):
            template = document_templates[i % len(document_templates)]
            text = template.format(
                crop=random.choice(self.crops),
                pest=random.choice(self.pests),
                season=random.choice(self.seasons),
                soil=random.choice(self.soil_types),
                fertilizer=random.choice(self.fertilizers),
                irrigation=random.choice(self.irrigation_types),
            )
            texts.append(text)
        
        return texts
    
    def generate_conversations(self, num_samples: int = 2000) -> List[List[Dict]]:
        """Generate synthetic conversation data"""
        
        conversations = []
        for i in range(num_samples):
            crop = random.choice(self.crops)
            pest = random.choice(self.pests)
            
            conv = [
                {"role": "user", "content": f"నమస్తే, నా {crop} పంటకు {pest} వచ్చింది. ఏమి చేయాలి?"},
                {"role": "assistant", "content": f"నమస్తే రైతు సోదరా! {crop} పంటకు {pest} వచ్చినట్లయితే, వెంటనే క్రింది చర్యలు తీసుకోండి:\n1. తెగులు గల మొక్కలను తొలగించండి\n2. సూచించిన మందులను వాడండి\n3. తరువాత నీటి నిర్వహణ జాగ్రత్తగా చూసుకోండి."},
                {"role": "user", "content": "ధన్యవాదాలు! ఏ మందులు వాడాలి?"},
                {"role": "assistant", "content": f"సాధారణంగా {random.choice(self.fertilizers)} వంటి సేంద్రీయ మందులు లేదా రసాయనిక మందులను వాడవచ్చు. అయితే స్థానిక వ్యవసాయ శాఖను సంప్రదించడం మంచిది."},
            ]
            conversations.append(conv)
        
        return conversations
    
    def save_data(self, data: List, filename: str, format: str = "json"):
        """Save data to file"""
        filepath = self.output_dir / filename
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format == "txt":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    if isinstance(item, str):
                        f.write(item + '\n\n')
                    else:
                        f.write(str(item) + '\n\n')
        
        print(f"Saved {len(data)} items to {filepath}")
    
    def collect_all(self):
        """Collect all types of data"""
        print("Generating Telugu agricultural dataset...")
        
        # Generate Q&A pairs
        print("Generating Q&A pairs...")
        qa_pairs = self.generate_qa_pairs(5000)
        self.save_data(qa_pairs, "qa_pairs.json", "json")
        
        # Generate agricultural documents
        print("Generating agricultural documents...")
        documents = self.generate_agricultural_text(10000)
        self.save_data(documents, "documents.txt", "txt")
        
        # Generate conversations
        print("Generating conversations...")
        conversations = self.generate_conversations(2000)
        self.save_data(conversations, "conversations.jsonl", "jsonl")
        
        # Create combined training corpus
        print("Creating combined corpus...")
        corpus_lines = []
        
        # Add documents
        for doc in documents:
            corpus_lines.append(doc)
        
        # Add Q&A formatted as text
        for qa in qa_pairs:
            text = f"ప్రశ్న: {qa['question']}\nసమాధానం: {qa['answer']}"
            corpus_lines.append(text)
        
        # Add conversations
        for conv in conversations:
            conv_text = []
            for msg in conv:
                if msg['role'] == 'user':
                    conv_text.append(f"వినియోగదారుడు: {msg['content']}")
                else:
                    conv_text.append(f"సహాయకుడు: {msg['content']}")
            corpus_lines.append('\n'.join(conv_text))
        
        # Save combined corpus
        corpus_file = self.output_dir / "training_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for line in corpus_lines:
                f.write(line + '\n\n')
        
        print(f"Combined corpus saved to {corpus_file}")
        print(f"Total lines: {len(corpus_lines)}")
        
        return {
            'qa_pairs': len(qa_pairs),
            'documents': len(documents),
            'conversations': len(conversations),
            'total_corpus_lines': len(corpus_lines),
        }


class DataProcessor:
    """Process raw data for training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
    
    def load_corpus(self, filename: str = "training_corpus.txt") -> str:
        """Load the training corpus"""
        filepath = self.raw_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_train_val_test(
        self,
        corpus_file: str = "training_corpus.txt",
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
    ):
        """Split corpus into train/val/test sets"""
        
        # Load corpus
        with open(self.raw_dir / corpus_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks (by double newline)
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        # Shuffle
        random.seed(42)
        random.shuffle(chunks)
        
        # Calculate split indices
        total = len(chunks)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_chunks = chunks[:train_end]
        val_chunks = chunks[train_end:val_end]
        test_chunks = chunks[val_end:]
        
        # Save splits
        splits = {
            'train': train_chunks,
            'val': val_chunks,
            'test': test_chunks,
        }
        
        for split_name, split_chunks in splits.items():
            output_file = self.processed_dir / f"{split_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for chunk in split_chunks:
                    f.write(chunk + '\n\n')
            print(f"Saved {len(split_chunks)} chunks to {output_file}")
        
        return splits


if __name__ == "__main__":
    # Collect data
    collector = TeluguAgriDataCollector()
    stats = collector.collect_all()
    print(f"\nData collection complete!")
    print(f"Stats: {stats}")
    
    # Process data
    processor = DataProcessor()
    processor.split_train_val_test()
