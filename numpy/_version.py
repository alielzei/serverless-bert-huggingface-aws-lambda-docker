import json
version_json = '\n{\n "date": "2023-06-25T20:31:30-0600",\n "dirty": false,\n "error": null,\n "full-revisionid": "9315a9072b2636f75c831b4eca9f42a5f67ca2fb",\n "version": "1.24.4"\n}\n'

def get_versions():
    return json.loads(version_json)

