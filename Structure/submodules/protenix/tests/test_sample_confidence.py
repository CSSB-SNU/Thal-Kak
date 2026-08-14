# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from protenix.model.sample_confidence import calculate_chain_pair_pae


class TestCalculateChainPairPAE(unittest.TestCase):
    def test_basic_two_chains(self):
        """Test basic case with two chains"""
        N_sample = 1
        N_token = 6
        
        token_pair_pae = torch.zeros(N_sample, N_token, N_token)
        token_pair_pae[0, :3, :3] = 1.0  # Chain 0 internal
        token_pair_pae[0, 3:, 3:] = 2.0  # Chain 1 internal
        token_pair_pae[0, :3, 3:] = 3.0  # Chain 0->1
        token_pair_pae[0, 3:, :3] = 4.0  # Chain 1->0
        
        asym_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        token_has_frame = torch.ones(N_token, dtype=torch.bool)
        
        result = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)
        
        self.assertIn("chain_pair_pae_mean", result)
        self.assertIn("chain_pair_pae_min", result)
        
        mean = result["chain_pair_pae_mean"]
        min_val = result["chain_pair_pae_min"]
        
        self.assertEqual(mean.shape, (N_sample, 2, 2))
        self.assertEqual(min_val.shape, (N_sample, 2, 2))
        
        # Check cross-chain PAE - the function computes separate 0->1 and 1->0
        # and stores them in respective positions (not symmetric)
        self.assertTrue(torch.allclose(mean[0, 0, 1], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(min_val[0, 0, 1], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(mean[0, 1, 0], torch.tensor(4.0)))
        self.assertTrue(torch.allclose(min_val[0, 1, 0], torch.tensor(4.0)))
    
    def test_with_contact_probs(self):
        """Test with custom contact probabilities"""
        N_sample = 1
        N_token = 4
        
        token_pair_pae = torch.zeros(N_sample, N_token, N_token)
        token_pair_pae[0, :2, 2:] = 5.0
        token_pair_pae[0, 2:, :2] = 10.0
        
        contact_probs = torch.zeros(N_token, N_token)
        contact_probs[:2, 2:] = 0.1
        contact_probs[2:, :2] = 0.9
        
        asym_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        token_has_frame = torch.ones(N_token, dtype=torch.bool)
        
        result = calculate_chain_pair_pae(
            token_pair_pae, asym_id, token_has_frame, contact_probs=contact_probs
        )
        
        mean = result["chain_pair_pae_mean"]
        
        # chain_pair_pae_mean[0, a, b] only uses contact_probs[a_mask, b_mask]
        # So 0->1 uses contact_probs[:2, 2:] = 0.1
        self.assertTrue(torch.allclose(mean[0, 0, 1], torch.tensor(5.0)))
        # 1->0 uses contact_probs[2:, :2] = 0.9
        self.assertTrue(torch.allclose(mean[0, 1, 0], torch.tensor(10.0)))
    
    def test_single_chain(self):
        """Test with only one chain"""
        N_sample = 1
        N_token = 4
        
        token_pair_pae = torch.ones(N_sample, N_token, N_token)
        asym_id = torch.zeros(N_token, dtype=torch.long)
        token_has_frame = torch.ones(N_token, dtype=torch.bool)
        
        result = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)
        
        mean = result["chain_pair_pae_mean"]
        min_val = result["chain_pair_pae_min"]
        
        self.assertEqual(mean.shape, (N_sample, 1, 1))
        self.assertEqual(min_val.shape, (N_sample, 1, 1))
    
    def test_no_valid_tokens(self):
        """Test with no tokens having frame"""
        N_sample = 1
        N_token = 4
        
        token_pair_pae = torch.ones(N_sample, N_token, N_token)
        asym_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        token_has_frame = torch.zeros(N_token, dtype=torch.bool)
        
        result = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)
        
        mean = result["chain_pair_pae_mean"]
        min_val = result["chain_pair_pae_min"]
        
        self.assertTrue(torch.isnan(mean[0, 0, 1]))
        self.assertTrue(torch.isnan(min_val[0, 0, 1]))
    
    def test_multiple_samples(self):
        """Test with multiple samples"""
        N_sample = 2
        N_token = 4
        
        token_pair_pae = torch.zeros(N_sample, N_token, N_token)
        token_pair_pae[0, :2, 2:] = 1.0
        token_pair_pae[0, 2:, :2] = 2.0
        token_pair_pae[1, :2, 2:] = 3.0
        token_pair_pae[1, 2:, :2] = 4.0
        
        asym_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        token_has_frame = torch.ones(N_token, dtype=torch.bool)
        
        result = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)
        
        mean = result["chain_pair_pae_mean"]
        
        self.assertTrue(torch.allclose(mean[0, 0, 1], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(mean[0, 1, 0], torch.tensor(2.0)))
        self.assertTrue(torch.allclose(mean[1, 0, 1], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(mean[1, 1, 0], torch.tensor(4.0)))
    
    def test_gapped_asym_id(self):
        """Test with non-contiguous asym_id"""
        N_sample = 1
        N_token = 4
        
        token_pair_pae = torch.ones(N_sample, N_token, N_token)
        asym_id = torch.tensor([1, 1, 3, 3], dtype=torch.long)  # Gapped IDs
        token_has_frame = torch.ones(N_token, dtype=torch.bool)
        
        result = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)
        
        mean = result["chain_pair_pae_mean"]
        self.assertEqual(mean.shape, (N_sample, 2, 2))  # Should remap to 0 and 1


class TestSampleConfidence(unittest.TestCase):
    def test_calculate_chain_pair_pae(self):
        """Existing test case preserved for backward compatibility"""
        N_sample = 1
        N_token = 6
        token_pair_pae = torch.ones(N_sample, N_token, N_token)
        token_pair_pae[0, :3, 3:] = 1.0
        token_pair_pae[0, 3:, :3] = 1.0
        asym_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        token_has_frame = torch.ones(N_token, dtype=torch.bool)
        
        result = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)
        chain_pair_pae_min = result["chain_pair_pae_min"]
        
        self.assertEqual(chain_pair_pae_min.shape, (N_sample, 2, 2))
        self.assertTrue(torch.allclose(chain_pair_pae_min[0, 0, 1], torch.tensor(1.0)))


if __name__ == "__main__":
    unittest.main()

