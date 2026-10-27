# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
import hashlib
import json

VERSION = "FlawPrint 1.1"

# Keep released settings available without external configuration files.
PRESETS = {
    'conversion': {
        'dataset': 'custom',
        'columns': {
            'id': 'news_id',
            'text': 'content',
            'image': 'image_path',
            'label': 'target',
            'domain': 'category',
            'group': 'event_id',
            'split': 'partition'
        },
        'labels': 'fake-one',
        'image_root': './data/images',
        'image_policy': 'error',
        'id_mode': 'source',
        'split_map': {
            'val': 'validation',
            'cal': 'calibration'
        },
        'exclude_domains': [],
        'keep_columns': []
    },
    'default': {
        'shrinkage': 0.1,
        'local_shrinkage': 0.35,
        'bootstrap': 100,
        'min_observed': 0.8,
        'min_samples': 20,
        'sign_tolerance': 0.03,
        'sign_weight': 1.0,
        'kappa': 2.0,
        'seed': 42
    },
    'experiment': {
        'method': 'FlawPrint',
        'datasets': ['weibo', 'weibo21', 'finefake'],
        'hardware': {
            'gpu': 'NVIDIA GeForce RTX 4090',
            'gpu_count': 1,
            'device': 'cuda:0'
        },
        'seeds': [42, 52, 3047, 3407],
        'partition_policy': {
            'mode': 'preserve_predefined',
            'required_roles': ['train', 'validation', 'test'],
            'optional_roles': ['calibration', 'profile'],
            'resplit_per_seed': False,
            'validation_use': 'Checkpoint selection and early stopping',
            'test_use': 'Final evaluation only',
            'calibration_use': 'Optional independent labeled source calibration',
            'profile_use': 'Optional independent unlabeled target reference'
        },
        'semantic_backbone': {
            'epochs': 50,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'max_text_length': 197,
            'early_stopping_patience': 5,
            'weight_decay': 5e-05,
            'embedding_dimension': 768
        },
        'partitions': {
            'weibo': {
                'train': './data/train_origin.csv',
                'validation': './data/val_origin.csv',
                'test': './data/test_origin.csv',
                'domains': 9,
                'input_labels': 'fake-one',
                'split_source': 'Existing prepared benchmark annotations'
            },
            'weibo21': {
                'train': './weibo21/train_datasets.xlsx',
                'validation': './weibo21/val_datasets.xlsx',
                'test': './weibo21/test_datasets.xlsx',
                'domains': 9,
                'input_labels': 'fake-one',
                'split_source': 'Existing prepared benchmark annotations'
            },
            'finefake': {
                'train': './FineFake/train.csv',
                'validation': './FineFake/val.csv',
                'test': './FineFake/test.csv',
                'domains': 7,
                'input_labels': 'real-one',
                'split_source': 'Existing label-stratified split files',
                'split_ratio': [0.6, 0.2, 0.2],
                'split_seed': 3407,
                'stratify': 'label'
            }
        },
        'metrics': [
            'accuracy',
            'fake_precision',
            'fake_recall',
            'fake_f1',
            'real_precision',
            'real_recall',
            'real_f1',
            'macro_f1',
            'auc'
        ],
        'neural_extension': {
            'name': 'FlawPrint-CRA',
            'entry_point': 'deep.py',
            'config': 'builtin:neural',
            'frozen_encoders': ['bert', 'mae', 'clip'],
            'optional_encoder': 'forensic',
            'modes': ['coupled', 'unbiased', 'independent', 'deep', 'concat', 'uniform', 'broadcast', 'diagonal', 'shuffled'],
            'source_reference': 'group cross-fitting',
            'target_reference': 'independent unlabeled profile rows',
            'target_labels_used_for_fitting': False,
            'relation_operator': 'content-conditioned-reference-v2',
            'fixed_checkpoint_intervention': 'deep.py predict --explain'
        }
    },
    'finefake': {
        'dataset': 'finefake',
        'labels': 'real-one',
        'image_root': '.',
        'image_policy': 'error',
        'id_mode': 'source'
    },
    'labels': {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1
    },
    'neural': {
        'dimension': 128,
        'heads': 4,
        'layers': 2,
        'dropout': 0.1,
        'bias_limit': 2.0,
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'weight_decay': 5e-05,
        'patience': 5,
        'seed': 42,
        'folds': 2,
        'auxiliary_weight': 0.1,
        'mode': 'coupled'
    },
    'quick': {
        'shrinkage': 0.15,
        'local_shrinkage': 0.35,
        'bootstrap': 12,
        'min_observed': 0.8,
        'min_samples': 12,
        'sign_tolerance': 0.03,
        'sign_weight': 1.0,
        'kappa': 2.0,
        'seed': 42
    },
    'reproduction': {
        'hardware': {
            'gpu': 'NVIDIA GeForce RTX 4090',
            'gpu_count': 1,
            'source': 'Author supplied'
        },
        'seeds': [42, 52, 3047, 3407],
        'configuration_files': ['builtin:seed42', 'builtin:seed52', 'builtin:seed3047', 'builtin:seed3407'],
        'path_base': 'Repository working directory',
        'physical_channel_device': 'cpu',
        'optional_semantic_device': 'cuda:0',
        'split_policy': 'Reuse frozen split identities across all training seeds'
    },
    'schema': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'FlawPrint canonical manifest row',
        'type': 'object',
        'required': ['id', 'image', 'text', 'domain', 'platform', 'group', 'split'],
        'properties': {
            'id': {
                'type': 'string',
                'minLength': 1
            },
            'image': {
                'type': 'string',
                'description': 'One local relative path or an empty string for text-only input'
            },
            'text': {
                'type': 'string'
            },
            'domain': {
                'type': 'string',
                'minLength': 1
            },
            'platform': {
                'type': 'string',
                'minLength': 1
            },
            'group': {
                'type': 'string',
                'minLength': 1
            },
            'split': {
                'enum': ['train', 'validation', 'calibration', 'profile', 'test', 'unspecified']
            },
            'label': {
                'type': 'integer',
                'enum': [0, 1],
                'description': 'Zero is real and one is fake'
            },
            'has_image': {
                'type': 'boolean'
            },
            'semantic_logit': {
                'type': 'number',
                'description': 'Finite fake-positive external score'
            },
            'patch_signs_valid_after_attack': {
                'type': 'boolean'
            }
        },
        'allOf': [
            {
                'if': {
                    'properties': {
                        'split': {
                            'const': 'profile'
                        }
                    },
                    'required': ['split']
                },
                'then': {
                    'not': {
                        'required': ['label']
                    }
                }
            }
        ],
        'additionalProperties': True
    },
    'seed3047': {
        'shrinkage': 0.1,
        'local_shrinkage': 0.35,
        'bootstrap': 100,
        'min_observed': 0.8,
        'min_samples': 20,
        'sign_tolerance': 0.03,
        'sign_weight': 1.0,
        'kappa': 2.0,
        'seed': 3047
    },
    'seed3407': {
        'shrinkage': 0.1,
        'local_shrinkage': 0.35,
        'bootstrap': 100,
        'min_observed': 0.8,
        'min_samples': 20,
        'sign_tolerance': 0.03,
        'sign_weight': 1.0,
        'kappa': 2.0,
        'seed': 3407
    },
    'seed42': {
        'shrinkage': 0.1,
        'local_shrinkage': 0.35,
        'bootstrap': 100,
        'min_observed': 0.8,
        'min_samples': 20,
        'sign_tolerance': 0.03,
        'sign_weight': 1.0,
        'kappa': 2.0,
        'seed': 42
    },
    'seed52': {
        'shrinkage': 0.1,
        'local_shrinkage': 0.35,
        'bootstrap': 100,
        'min_observed': 0.8,
        'min_samples': 20,
        'sign_tolerance': 0.03,
        'sign_weight': 1.0,
        'kappa': 2.0,
        'seed': 52
    },
    'sources': {
        'verified_date': '2026-10-09',
        'papers': [
            {
                'id': 'FineFake',
                'title': 'FineFake: A Knowledge-Enriched Dataset for Fine-Grained Multi-Domain Fake News Detection',
                'year': 2024,
                'role': 'Dataset and cross-domain evaluation protocol',
                'page': 'https://arxiv.org/abs/2404.01336v3',
                'pdf': 'https://arxiv.org/pdf/2404.01336v3',
                'file': 'FineFake.pdf',
                'repository': 'https://github.com/Accuser907/FineFake',
                'implementation': 'External official dataset; local manifest converter'
            },
            {
                'id': 'MDFEND',
                'title': 'MDFEND: Multi-domain Fake News Detection',
                'year': 2021,
                'role': 'Domain-gated semantic comparison and Weibo21',
                'page': 'https://arxiv.org/abs/2201.00987v1',
                'pdf': 'https://arxiv.org/pdf/2201.00987v1',
                'file': 'MDFEND.pdf',
                'repository': 'https://github.com/kennqiang/MDFEND-Weibo21',
                'implementation': 'External official predictions supported'
            },
            {
                'id': 'M3FEND',
                'title': 'Memory-Guided Multi-View Multi-Domain Fake News Detection',
                'year': 2022,
                'role': 'Semantic multi-view and domain memory comparison',
                'page': 'https://arxiv.org/abs/2206.12808v1',
                'pdf': 'https://arxiv.org/pdf/2206.12808v1',
                'file': 'M3FEND.pdf',
                'repository': 'https://github.com/ICTMCG/M3FEND',
                'implementation': 'External official predictions supported'
            },
            {
                'id': 'MMDFND',
                'title': 'MMDFND: Multi-modal Multi-Domain Fake News Detection',
                'year': 2024,
                'role': 'Multimodal multi-domain semantic comparison',
                'page': 'https://dl.acm.org/doi/10.1145/3664647.3681317',
                'pdf': 'https://openreview.net/pdf?id=sdF3MuyHtz',
                'fallback_pdf': [
                    'https://api2.openreview.net/pdf?id=sdF3MuyHtz',
                    'https://api.openreview.net/pdf?id=sdF3MuyHtz',
                    'https://openreview.net/pdf/48770420923e953f48222c349b757195120648f6.pdf',
                    'https://dl.acm.org/doi/pdf/10.1145/3664647.3681317'
                ],
                'file': 'MMDFND.pdf',
                'repository': 'https://github.com/yutchina/MMDFND',
                'implementation': 'External official predictions supported'
            },
            {
                'id': 'UniversalFakeDetect',
                'title': 'Towards Universal Fake Image Detectors that Generalize Across Generative Models',
                'year': 2023,
                'role': 'Cross-generator image detection comparison',
                'page': 'https://arxiv.org/abs/2302.10174v1',
                'pdf': 'https://arxiv.org/pdf/2302.10174v1',
                'file': 'UniversalFakeDetect.pdf',
                'repository': 'https://github.com/WisconsinAIVision/UniversalFakeDetect',
                'implementation': 'External official predictions supported; optional local CLIP control is distinct'
            },
            {
                'id': 'DIRE',
                'title': 'DIRE for Diffusion-Generated Image Detection',
                'year': 2023,
                'role': 'Diffusion reconstruction forensic comparison',
                'page': 'https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DIRE_for_Diffusion-Generated_Image_Detection_ICCV_2023_paper.html',
                'pdf': 'https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DIRE_for_Diffusion-Generated_Image_Detection_ICCV_2023_paper.pdf',
                'file': 'DIRE.pdf',
                'repository': 'https://github.com/ZhendongWang6/DIRE',
                'implementation': 'External official predictions supported'
            },
            {
                'id': 'DAMMFND',
                'title': 'DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection',
                'year': 2025,
                'role': 'Domain-aware multimodal multi-view comparison',
                'page': 'https://ojs.aaai.org/index.php/AAAI/article/view/32036',
                'pdf': 'https://ojs.aaai.org/index.php/AAAI/article/download/32036/34191',
                'file': 'DAMMFND.pdf',
                'repository': 'https://github.com/luweihai/DAMMFND',
                'implementation': 'External official predictions supported'
            },
            {
                'id': 'ConceptShift',
                'title': 'Early Detection of Misinformation for Infodemic Management: A Domain Adaptation Approach',
                'year': 2024,
                'role': 'Concept shift background',
                'page': 'https://arxiv.org/abs/2406.10238v1',
                'pdf': 'https://arxiv.org/pdf/2406.10238v1',
                'file': 'ConceptShift.pdf',
                'repository': None,
                'implementation': 'Background reference only'
            },
            {
                'id': 'ModalityBias',
                'title': 'Unveiling Modality Bias: Automated Sample-Specific Analysis for Multimodal Misinformation Benchmarks',
                'year': 2025,
                'role': 'Modality attribution background',
                'page': 'https://arxiv.org/abs/2511.05883v1',
                'pdf': 'https://arxiv.org/pdf/2511.05883v1',
                'file': 'ModalityBias.pdf',
                'repository': None,
                'implementation': 'Background reference only'
            },
            {
                'id': 'ConformalPrediction',
                'title': 'A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification',
                'year': 2021,
                'role': 'Labeled split conformal theory',
                'page': 'https://arxiv.org/abs/2107.07511v6',
                'pdf': 'https://arxiv.org/pdf/2107.07511v6',
                'file': 'ConformalPrediction.pdf',
                'repository': None,
                'implementation': 'Local labeled split conformal component'
            },
            {
                'id': 'Sinkhorn',
                'title': 'Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances',
                'year': 2013,
                'role': 'Entropic transport theory',
                'page': 'https://arxiv.org/abs/1306.0895v1',
                'pdf': 'https://arxiv.org/pdf/1306.0895v1',
                'file': 'Sinkhorn.pdf',
                'repository': None,
                'implementation': 'Local log-domain transport solver'
            },
            {
                'id': 'DAPTReference',
                'title': "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks",
                'year': 2020,
                'role': '2020 domain-adaptive pretraining background; distinct from DAPT domain-aware multimodal fake news detection (ACM MM 2025)',
                'page': 'https://aclanthology.org/2020.acl-main.740/',
                'pdf': 'https://aclanthology.org/2020.acl-main.740.pdf',
                'file': 'DAPTReference.pdf',
                'repository': None,
                'implementation': 'Background mechanism comparison; external predictions supported'
            },
            {
                'id': 'DANN',
                'title': 'Domain-Adversarial Training of Neural Networks',
                'year': 2016,
                'role': 'Gradient-reversal domain adaptation mechanism',
                'page': 'https://www.jmlr.org/beta/papers/v17/15-239.html',
                'pdf': 'https://jmlr.org/papers/volume17/15-239/15-239.pdf',
                'file': 'DANN.pdf',
                'repository': None,
                'implementation': 'Background mechanism comparison; no local official reproduction'
            },
            {
                'id': 'ConceptShiftV2',
                'title': 'Early Detection of Misinformation for Infodemic Management: A Domain Adaptation Approach',
                'year': 2026,
                'role': 'Updated DACA concept alignment reference for abstract and novelty framing',
                'page': 'https://arxiv.org/abs/2406.10238v2',
                'pdf': 'https://arxiv.org/pdf/2406.10238v2',
                'file': 'ConceptShift_v2.pdf',
                'repository': None,
                'implementation': 'Background mechanism comparison; v1 retained separately'
            },
            {
                'id': 'MediaLifecycle',
                'title': 'Trustworthy synthetic visual media: Evidence across the media lifecycle',
                'year': 2026,
                'role': 'Media lifecycle evidence and processing conditions',
                'page': 'https://arxiv.org/abs/2609.34232v1',
                'pdf': 'https://arxiv.org/pdf/2609.34232v1',
                'file': 'MediaLifecycle.pdf',
                'repository': None,
                'implementation': 'Background reference; no first-use claim for dependency modeling'
            }
        ],
        'additional_writing_references': [
            {
                'id': 'DAPT',
                'title': 'DAPT: Domain-Aware Prompt-Tuning for Multimodal Fake News Detection',
                'year': 2025,
                'status': 'ACM MM 2025',
                'page': 'https://doi.org/10.1145/3746027.3755192',
                'pdf': 'https://dl.acm.org/doi/pdf/10.1145/3746027.3755192',
                'file': 'DAPT.pdf',
                'role': 'Direct domain-aware comparison and writing model',
                'date': None,
                'authors': ['Yu Tong', 'Weihai Lu', 'Xiaoxi Cui', 'Yifan Mao', 'Zhejun Zhao'],
                'doi': '10.1145/3746027.3755192',
                'bibliographic_metadata_source': 'https://yutchina.github.io/',
                'date_basis': 'Official publication date or conference month when available',
                'download_status': 'pending publisher access',
                'local_library': '../writing/references/README.md',
                'implementation': 'Writing reference; no local official training reproduction'
            }
        ],
        'writing_references_updated_on': '2026-10-12'
    },
    'weibo': {
        'dataset': 'weibo',
        'labels': 'fake-one',
        'image_root': '.',
        'image_policy': 'error',
        'id_mode': 'source'
    },
    'weibo21': {
        'dataset': 'weibo21',
        'labels': 'fake-one',
        'image_root': '.',
        'image_policy': 'error',
        'id_mode': 'source'
    }
}

def is_builtin(value: object) -> bool:
    return isinstance(value, str) and value.startswith("builtin:")

def configuration(value: str) -> dict:
    if is_builtin(value):
        name = value.removeprefix("builtin:")
        if name not in PRESETS:
            raise ValueError("Unknown built-in configuration: " + name)
        # Copy presets so callers cannot change later runs.
        return copy.deepcopy(PRESETS[name])
    from engine.storage import read_json
    result = read_json(value)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a JSON object")
    return result

def configuration_record(value: str) -> dict:
    if is_builtin(value):
        encoded = json.dumps(configuration(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return {"preset": value, "sha256": hashlib.sha256(encoded).hexdigest()}
    from engine.storage import digest
    return {"path": value, "sha256": digest(value)}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
