"""
Evaluation Module for Telugu Agricultural SLM
"""
import os
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

import torch
from tqdm import tqdm

from model import TeluguAgriSLM
from tokenizer import TeluguAgriTokenizer
from inference import TeluguAgriInference


class Evaluator:
    """Evaluator for Telugu Agricultural SLM"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: Optional[str] = None,
    ):
        self.inference = TeluguAgriInference(model_path, tokenizer_path, device)
        self.device = self.inference.device
        
        # Load test data
        self.test_data = self._load_test_data()
    
    def _load_test_data(self) -> List[Dict]:
        """Load test dataset"""
        test_file = "data/processed/test.txt"
        if not os.path.exists(test_file):
            print(f"Warning: Test file not found at {test_file}")
            return []
        
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse test examples
        examples = []
        chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        
        for chunk in chunks[:100]:  # Limit to 100 test examples
            # Try to extract Q&A
            if 'ప్రశ్న:' in chunk and 'సమాధానం:' in chunk:
                parts = chunk.split('సమాధానం:')
                if len(parts) == 2:
                    question = parts[0].replace('ప్రశ్న:', '').strip()
                    answer = parts[1].strip()
                    examples.append({
                        'question': question,
                        'reference': answer,
                    })
        
        return examples
    
    def calculate_perplexity(self, text_file: str, max_samples: int = 1000) -> float:
        """Calculate perplexity on a text file"""
        print(f"Calculating perplexity on {text_file}...")
        
        # Load text
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        chunks = chunks[:max_samples]
        
        total_loss = 0.0
        total_tokens = 0
        
        self.inference.model.eval()
        
        for chunk in tqdm(chunks, desc="Calculating perplexity"):
            # Encode
            input_ids = self.inference.tokenizer.encode(chunk, add_special_tokens=True)
            
            if len(input_ids) < 2:
                continue
            
            # Truncate if too long
            if len(input_ids) > 512:
                input_ids = input_ids[:512]
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.inference.model(input_tensor, labels=input_tensor)
                loss = outputs['loss']
            
            total_loss += loss.item() * (len(input_ids) - 1)
            total_tokens += len(input_ids) - 1
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def evaluate_qa_accuracy(self, num_samples: Optional[int] = None) -> Dict:
        """Evaluate Q&A accuracy"""
        if not self.test_data:
            print("No test data available")
            return {}
        
        test_samples = self.test_data[:num_samples] if num_samples else self.test_data
        
        print(f"Evaluating Q&A accuracy on {len(test_samples)} samples...")
        
        correct = 0
        results = []
        
        for sample in tqdm(test_samples, desc="Evaluating Q&A"):
            question = sample['question']
            reference = sample['reference']
            
            # Generate answer
            predicted = self.inference.answer_question(question)
            
            # Simple keyword matching for accuracy
            is_correct = self._check_answer_correctness(predicted, reference)
            
            if is_correct:
                correct += 1
            
            results.append({
                'question': question,
                'reference': reference,
                'predicted': predicted,
                'correct': is_correct,
            })
        
        accuracy = correct / len(test_samples) if test_samples else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_samples),
            'results': results,
        }
    
    def _check_answer_correctness(self, predicted: str, reference: str) -> bool:
        """Check if predicted answer matches reference (simple heuristic)"""
        # Extract keywords (Telugu words longer than 2 chars)
        pred_words = set(re.findall(r'[\u0C00-\u0C7F]{2,}', predicted))
        ref_words = set(re.findall(r'[\u0C00-\u0C7F]{2,}', reference))
        
        if not ref_words:
            return False
        
        # Calculate overlap
        overlap = len(pred_words & ref_words)
        overlap_ratio = overlap / len(ref_words)
        
        return overlap_ratio >= 0.3  # At least 30% word overlap
    
    def evaluate_agricultural_knowledge(self) -> Dict:
        """Evaluate agricultural domain knowledge"""
        
        # Test questions covering different agricultural topics
        test_questions = [
            {
                "category": "crop_management",
                "question": "వరి పంట ఎప్పుడు వేయాలి?",
                "keywords": ["ఖరీఫ్", "రబీ", "వర్షాకాలం", "నాట్లు"],
            },
            {
                "category": "pest_control",
                "question": "పత్తి పంటలో తెగులు నివారణ ఎలా?",
                "keywords": ["మందు", "తెగులు", "పురుగు", "స్ప్రే"],
            },
            {
                "category": "fertilizers",
                "question": "ఎరువుల వాడకం ఎలా ఉండాలి?",
                "keywords": ["యూరియా", "డీఏపీ", "ఎరువు", "నేల"],
            },
            {
                "category": "irrigation",
                "question": "నీటిపారుదల పద్ధతులు ఏమిటి?",
                "keywords": ["నీరు", "త్రాగు", "పారుదల", "వ్యవస్థ"],
            },
            {
                "category": "soil",
                "question": "నేల సారాన్ని ఎలా పెంచాలి?",
                "keywords": ["సేంద్రీయ", "కంపోస్ట్", "నేల", "సారం"],
            },
        ]
        
        results = defaultdict(list)
        
        for test in test_questions:
            question = test['question']
            keywords = test['keywords']
            category = test['category']
            
            # Generate answer
            answer = self.inference.answer_question(question)
            
            # Check for keyword presence
            keyword_hits = sum(1 for kw in keywords if kw in answer)
            coverage = keyword_hits / len(keywords)
            
            results[category].append({
                'question': question,
                'answer': answer,
                'keyword_coverage': coverage,
                'keywords_found': keyword_hits,
                'total_keywords': len(keywords),
            })
        
        # Calculate category scores
        category_scores = {}
        for category, items in results.items():
            avg_coverage = sum(item['keyword_coverage'] for item in items) / len(items)
            category_scores[category] = avg_coverage
        
        return {
            'category_scores': dict(category_scores),
            'overall_score': sum(category_scores.values()) / len(category_scores),
            'detailed_results': dict(results),
        }
    
    def evaluate_telugu_language(self) -> Dict:
        """Evaluate Telugu language understanding and generation"""
        
        test_cases = [
            {
                "name": "telugu_script_preservation",
                "prompt": "వరి పంట గురించి వివరించండి:",
                "check": lambda text: any('\u0C00' <= c <= '\u0C7F' for c in text),
            },
            {
                "name": "coherent_response",
                "prompt": "మొక్కజొన్న సాగు విధానం ఏమిటి?",
                "check": lambda text: len(text) > 20 and '.' in text,
            },
            {
                "name": "question_answering",
                "prompt": "ప్రశ్న: పత్తి ఎప్పుడు కోయాలి?\nసమాధానం:",
                "check": lambda text: len(text) > 10,
            },
        ]
        
        results = []
        passed = 0
        
        for test in test_cases:
            response = self.inference.generate(test['prompt'])
            success = test['check'](response)
            
            if success:
                passed += 1
            
            results.append({
                'name': test['name'],
                'prompt': test['prompt'],
                'response': response,
                'passed': success,
            })
        
        return {
            'total_tests': len(test_cases),
            'passed': passed,
            'failed': len(test_cases) - passed,
            'success_rate': passed / len(test_cases),
            'results': results,
        }
    
    def run_full_evaluation(self, output_file: Optional[str] = None) -> Dict:
        """Run full evaluation suite"""
        print("=" * 60)
        print("Telugu Agricultural SLM - Full Evaluation")
        print("=" * 60)
        
        results = {}
        
        # 1. Perplexity evaluation
        print("\n1. Calculating Perplexity...")
        test_file = "data/processed/test.txt"
        if os.path.exists(test_file):
            perplexity = self.calculate_perplexity(test_file)
            results['perplexity'] = perplexity
            print(f"   Perplexity: {perplexity:.2f}")
        
        # 2. Q&A Accuracy
        print("\n2. Evaluating Q&A Accuracy...")
        qa_results = self.evaluate_qa_accuracy(num_samples=50)
        results['qa_accuracy'] = qa_results
        if qa_results:
            print(f"   Accuracy: {qa_results['accuracy']:.2%}")
            print(f"   Correct: {qa_results['correct']}/{qa_results['total']}")
        
        # 3. Agricultural Knowledge
        print("\n3. Evaluating Agricultural Knowledge...")
        agri_results = self.evaluate_agricultural_knowledge()
        results['agricultural_knowledge'] = agri_results
        print(f"   Overall Score: {agri_results['overall_score']:.2%}")
        for cat, score in agri_results['category_scores'].items():
            print(f"   - {cat}: {score:.2%}")
        
        # 4. Telugu Language
        print("\n4. Evaluating Telugu Language...")
        lang_results = self.evaluate_telugu_language()
        results['telugu_language'] = lang_results
        print(f"   Success Rate: {lang_results['success_rate']:.2%}")
        print(f"   Passed: {lang_results['passed']}/{lang_results['total_tests']}")
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {output_path}")
        
        # Summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        if 'perplexity' in results:
            print(f"Perplexity: {results['perplexity']:.2f}")
        if 'qa_accuracy' in results and results['qa_accuracy']:
            print(f"Q&A Accuracy: {results['qa_accuracy']['accuracy']:.2%}")
        print(f"Agri Knowledge: {results['agricultural_knowledge']['overall_score']:.2%}")
        print(f"Telugu Language: {results['telugu_language']['success_rate']:.2%}")
        
        return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Telugu Agricultural SLM")
    parser.add_argument("--model", default="models/checkpoints/final/pytorch_model.bin",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="models/tokenizer/telugu_agri_spm.model",
                       help="Path to tokenizer")
    parser.add_argument("--output", default="models/evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        print("Please train the model first.")
        return
    
    # Run evaluation
    evaluator = Evaluator(args.model, args.tokenizer)
    results = evaluator.run_full_evaluation(args.output)


if __name__ == "__main__":
    main()
