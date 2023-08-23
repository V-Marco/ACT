import torch


def correlation_score(target_data: torch.Tensor, simulated_data: torch.Tensor) -> float:
    cov = (target_data - torch.mean(target_data, dim=1, keepdim=True)) * (
        simulated_data - torch.mean(simulated_data, dim=1, keepdim=True)
    )
    cov = torch.sum(cov, dim=1)

    var0 = torch.sum(
        torch.square(target_data - torch.mean(target_data, dim=1, keepdim=True)), dim=1
    )
    var1 = torch.sum(
        torch.square(simulated_data - torch.mean(simulated_data, dim=1, keepdim=True)),
        dim=1,
    )
    corr = cov / (torch.sqrt(var0 * var1) + 1e-15)

    return float(torch.mean(corr).detach())


def mse_score(target_data: torch.Tensor, simulated_data: torch.Tensor) -> float:
    return float(
        torch.mean(
            torch.mean(torch.square(target_data - simulated_data), dim=1)
        ).detach()
    )
