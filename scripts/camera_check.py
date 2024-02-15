import imagingcontrol4 as ic4
def print_device_list():
    print("Enumerating all attached video capture devices...")

    device_list = ic4.DeviceEnum.devices()

    if len(device_list) == 0:
        print("No devices found")
        return

    print(f"Found {len(device_list)} devices:")

    for device_info in device_list:
        print(format_device_info(device_info))

    def format_device_info(device_info: ic4.DeviceInfo) -> str:
        return f"Model: {device_info.model_name} Serial: {device_info.serial}"
    
if __name__ == "__main__":
    ic4.Library.init()
    print_device_list()