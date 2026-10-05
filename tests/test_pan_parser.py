import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from src.preprocessing.pan_parser import PANParser

class TestPANParser(unittest.TestCase):
    def test_parse_xml_file(self):
        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
<document reference="suspicious-document00007.txt">
  <feature name="about" authors="Test Author" title="Test Title" lang="en" />
  <feature name="plagiarism" type="artificial" obfuscation="low" this_language="en" this_offset="224" this_length="2807" source_reference="source-document06022.txt" source_language="en" source_offset="729730" source_length="2777" />
</document>
"""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            xml_file = tmp_path / "suspicious-document00007.xml"
            with open(xml_file, "w", encoding="utf-8") as f:
                f.write(sample_xml)

            parser = PANParser(tmpdir)
            annots, errors = parser.parse_xml_file(xml_file)

            self.assertEqual(len(errors), 0)
            self.assertEqual(len(annots), 1)
            ann = annots[0]
            self.assertEqual(ann.suspicious_id, "suspicious-document00007.txt")
            self.assertEqual(ann.source_id, "source-document06022.txt")
            self.assertEqual(ann.suspicious_offset, 224)
            self.assertEqual(ann.suspicious_length, 2807)
            self.assertEqual(ann.source_offset, 729730)
            self.assertEqual(ann.source_length, 2777)
            self.assertEqual(ann.obfuscation, "low")

if __name__ == "__main__":
    unittest.main()
