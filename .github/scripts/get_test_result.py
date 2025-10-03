#!/usr/bin/env python3
import xml.etree.ElementTree as ET


def main(report_path='report.xml'):
    try:
        tree = ET.parse(report_path)
        root = tree.getroot()
    except Exception as e:
        return 1

    failures = 0
    errors = 0
    if root.tag == 'testsuites':
        for ts in root.findall('testsuite'):
            failures += int(ts.attrib.get('failures', '0'))
            errors += int(ts.attrib.get('errors', '0'))
    elif root.tag == 'testsuite':
        failures = int(root.attrib.get('failures', '0'))
        errors = int(root.attrib.get('errors', '0'))
    else:
        for tc in root.findall('.//testcase'):
            if tc.find('failure') is not None:
                failures += 1
            if tc.find('error') is not None:
                errors += 1

    if failures > 0 or errors > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    exit(main())
