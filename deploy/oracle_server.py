"""Сервер Сірка на безкоштовному Oracle Cloud (ARM A1): знаходить його або створює.

Безкоштовні сервери ARM часто зайняті («Out of host capacity»), тож кожен запуск пробує всі
зони доступності: спершу 2 ядра й 12 ГБ (уся безкоштовна межа), тоді 1 ядро й 6 ГБ. Сервер
отримує щотижневі безкоштовні резервні копії диска. Адресу пише в GITHUB_OUTPUT (ip=…).
Доступ – з оточення: OCI_USER, OCI_TENANCY, OCI_FINGERPRINT, OCI_KEY, SERVER_AUTHORIZED_KEYS.
"""
import os
import sys
import time

import oci

NAME = "sirko-server"
REGION = "eu-frankfurt-1"
SHAPE = "VM.Standard.A1.Flex"
SIZES = ((2, 12), (1, 6))
BACKUP_POLICY = "weekly-free"


def out(key, value):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def main():
    config = {
        "user": os.environ["OCI_USER"],
        "tenancy": os.environ["OCI_TENANCY"],
        "fingerprint": os.environ["OCI_FINGERPRINT"],
        "key_content": os.environ["OCI_KEY"],
        "region": REGION,
    }
    oci.config.validate_config(config)
    tenancy = config["tenancy"]
    compute = oci.core.ComputeClient(config)
    network = oci.core.VirtualNetworkClient(config)
    storage = oci.core.BlockstorageClient(config)
    identity = oci.identity.IdentityClient(config)
    every = oci.pagination.list_call_get_all_results

    alive = ("PROVISIONING", "STARTING", "RUNNING", "STOPPING", "STOPPED")
    found = [i for i in every(compute.list_instances, tenancy, display_name=NAME).data if i.lifecycle_state in alive]
    if found:
        inst = found[0]
        print(f"Сервер уже є: {inst.shape_config.ocpus:g} ×ARM, {inst.shape_config.memory_in_gbs:g} ГБ, {inst.lifecycle_state}")
    else:
        image = next(
            img for img in every(
                compute.list_images, tenancy, operating_system="Canonical Ubuntu", operating_system_version="24.04",
                shape=SHAPE, sort_by="TIMECREATED", sort_order="DESC").data
            if "aarch64" in img.display_name and "Minimal" not in img.display_name)
        subnet = next(s for s in every(network.list_subnets, tenancy).data
                      if s.lifecycle_state == "AVAILABLE" and not s.prohibit_public_ip_on_vnic)
        ads = [ad.name for ad in identity.list_availability_domains(tenancy).data]
        inst = None
        for ocpus, mem in SIZES:
            for ad in ads:
                details = oci.core.models.LaunchInstanceDetails(
                    compartment_id=tenancy, availability_domain=ad, display_name=NAME, shape=SHAPE,
                    shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(ocpus=ocpus, memory_in_gbs=mem),
                    source_details=oci.core.models.InstanceSourceViaImageDetails(image_id=image.id),
                    create_vnic_details=oci.core.models.CreateVnicDetails(
                        subnet_id=subnet.id, assign_public_ip=True, hostname_label="sirko"),
                    metadata={"ssh_authorized_keys": os.environ["SERVER_AUTHORIZED_KEYS"].strip()})
                try:
                    inst = compute.launch_instance(details).data
                except oci.exceptions.ServiceError as e:
                    print(f"{ocpus} ×ARM, {mem} ГБ, {ad.split(':')[-1]}: {e.message}")
                    if e.status == 429:
                        time.sleep(30)
                    elif "capacity" not in e.message.lower():
                        raise
                    continue
                print(f"Створено сервер: {ocpus} ×ARM, {mem} ГБ, {ad.split(':')[-1]}, {image.display_name}")
                break
            if inst:
                break
        if not inst:
            print("Місця в Oracle поки немає, наступна спроба – за розкладом")
            out("ready", "false")
            return 0

    inst = oci.wait_until(compute, compute.get_instance(inst.id), "lifecycle_state", "RUNNING",
                          max_wait_seconds=900).data
    vnic_id = every(compute.list_vnic_attachments, tenancy, instance_id=inst.id).data[0].vnic_id
    ip = network.get_vnic(vnic_id).data.public_ip
    boot = every(compute.list_boot_volume_attachments, inst.availability_domain, tenancy, instance_id=inst.id).data[0]
    policy = next((p for p in every(storage.list_volume_backup_policies, compartment_id=tenancy).data
                   if p.display_name == BACKUP_POLICY), None)
    if policy and not storage.get_volume_backup_policy_asset_assignment(boot.boot_volume_id).data:
        storage.create_volume_backup_policy_assignment(oci.core.models.CreateVolumeBackupPolicyAssignmentDetails(
            asset_id=boot.boot_volume_id, policy_id=policy.id))
        print(f"Резервні копії диска: {BACKUP_POLICY}")
    out("ready", "true")
    out("ip", ip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
