import json
import os
import unittest
import numpy as np

from protenix.data.template.template_utils import TemplateHitFeaturizer
from protenix.data.constants import ATOM37_NUM

class TestJsonTemplateParser(unittest.TestCase):
    def test_json_template_parser(self):
        # Load the JSON file
        json_path = "examples/example_with_json_template/demo_ab.json"
        self.assertTrue(os.path.exists(json_path), f"File {json_path} does not exist")
        
        with open(json_path, "r") as f:
            data = json.load(f)
            
        # Extract the sequence and template list
        protein_chain = data[0]["sequences"][0]["proteinChain"]
        query_sequence = protein_chain["sequence"]
        template_path = protein_chain["templatesPath"]
        
        self.assertTrue(len(query_sequence) > 0)
        
        # Instantiate the featurizer
        featurizer = TemplateHitFeaturizer(
            mmcif_dir="/tmp/dummy_mmcif_dir",
            template_cache_dir=None,
            kalign_binary_path=None,
            _zero_center_positions=True,
        )

        with open(template_path, "r") as f:
            template_list = json.load(f)
        
        # Call the parse_json_templates method
        result = featurizer.parse_json_templates(
            template_list=template_list,
            query_sequence=query_sequence
        )
        
        # Assertions
        self.assertEqual(len(result.errors), 0, f"Expected no errors, but got: {result.errors}")
        self.assertEqual(len(result.features), len(template_list))
        self.assertEqual(len(result.hits), len(template_list))
        
        num_query = len(query_sequence)
        
        for i, feature in enumerate(result.features):
            self.assertIn("template_all_atom_positions", feature)
            self.assertIn("template_all_atom_masks", feature)
            self.assertIn("template_aatype", feature)
            self.assertIn("template_sequence", feature)
            self.assertIn("template_domain_names", feature)
            self.assertIn("template_sum_probs", feature)
            self.assertIn("template_release_date", feature)
            
            pos = feature["template_all_atom_positions"]
            mask = feature["template_all_atom_masks"]
            aatype = feature["template_aatype"]
            
            # Check shapes
            self.assertEqual(pos.shape, (num_query, ATOM37_NUM, 3), f"Expected pos shape {(num_query, ATOM37_NUM, 3)}, got {pos.shape}")
            self.assertEqual(mask.shape, (num_query, ATOM37_NUM), f"Expected mask shape {(num_query, ATOM37_NUM)}, got {mask.shape}")
            self.assertEqual(aatype.shape, (num_query,), f"Expected aatype shape {(num_query,)}, got {aatype.shape}")
            
            # Check if some masks are valid (not all zeros)
            self.assertTrue(np.sum(mask) > 0, "Expected atom masks to contain non-zero values")
            
            # Check hit information
            hit = result.hits[i]
            self.assertEqual(hit.query, query_sequence)
            self.assertEqual(hit.aligned_cols, len(template_list[i]["queryIndices"]))
            self.assertEqual(len(hit.indices_query), num_query)
            self.assertEqual(len(hit.indices_hit), num_query)

if __name__ == "__main__":
    unittest.main()
