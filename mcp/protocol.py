import uuid

def create_mcp_message(sender, receiver, msg_type, payload, trace_id=None):
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id or str(uuid.uuid4()),
        "payload": payload
    }
