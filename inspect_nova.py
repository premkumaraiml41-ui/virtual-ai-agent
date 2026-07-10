import boto3

client = boto3.client('nova-act', region_name='us-east-1')
print('methods:')
print([m for m in dir(client) if m[0].islower() and not m.startswith('_')])
print('\noperations:')
print(client.meta.service_model.operation_names)

for op_name in ['CreateWorkflowRun', 'CreateSession', 'InvokeActStep']:
    print(f'\n=== {op_name} ===')
    op_model = client.meta.service_model.operation_model(op_name)
    if op_model.input_shape:
        print('Input members:')
        for name, member in op_model.input_shape.members.items():
            member_names = list(member.members.keys()) if getattr(member, 'members', None) else None
            print(' ', name, member.type_name, member_names)
    else:
        print('No input shape')
    if op_model.output_shape:
        print('Output members:')
        for name, member in op_model.output_shape.members.items():
            member_names = list(member.members.keys()) if getattr(member, 'members', None) else None
            print(' ', name, member.type_name, member_names)
    else:
        print('No output shape')
